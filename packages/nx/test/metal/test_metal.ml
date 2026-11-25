(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal on the machine's Metal GPU 0, skipping without one: its copies' bits
   around a destination and between outputs that read each other's destinations,
   through workspace kept and of its own, in one launch or in pieces; its kept
   constants and submissions; and, through the harness, the ticket protocol's
   last group from 1 to 128 groups. Without the GPU, each launch its plans make
   keeps within nx_metal.mli's time per launch, a scan within the elements it
   keeps is planned at any length a slice, and a contraction's tiles, ranges and
   windows follow contract.mli's rule. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module S = Nx_metal_support
module Support = Nx_kernels_support
module Plan = Nx_metal.Plan

let answer =
  Testable.make ~pp:Nx_array_support.pp_answer ~equal:(fun a b -> a = b)

let random_bytes st n =
  String.init n (fun _ -> Char.chr (Random.State.int st 256))

(* A fresh array of [dt] and shape [s] on the GPU, its bytes drawn from [st]. *)
let drawn st dt s =
  let x = A.create (S.device ()) dt (Iarray.of_array s) in
  let b = A.buffer x in
  let n = Rig.Buffer.length b in
  Rig.Buffer.blit_from_string (random_bytes st n) 0 b 0 n;
  x

let elements x = Support.elements (A.Any x)

let backend () =
  match Support.metal () with
  | Some b -> b
  | None -> skip ~reason:"no Metal GPU nx.metal computes on" ()

let cost ?(bytes = 0) ?(ops = 0) ?(chain = 0) () = { Plan.bytes; ops; chain }

(* The planners, over the facts of a call of [dsts], [loads] and [carry]. *)
let facts ?carry ~dsts loads = Nx_metal.Operands.v ?carry ~dsts loads

let loop_map ~holds p ~dsts loads =
  Nx_metal.Loop.map ~holds p (facts ~dsts loads)

let loop_reduce ~holds p ~shape ~axes rs ~dsts loads =
  Nx_metal.Loop.reduce ~holds p ~shape ~axes rs (facts ~dsts loads)

let loop_scan ?carry ~holds p ~axis rk ~dsts loads =
  Nx_metal.Loop.scan ~holds p ~axis rk (facts ?carry ~dsts loads)

let moves_map p ~dsts loads = Nx_metal.Moves.map p (facts ~dsts loads)

(* Copies *)

type around = { dt : int; before : int; count : int; after : int; seed : int }

let pp_around ppf c =
  let (D.Any dt) = List.nth D.all c.dt in
  Format.fprintf ppf "%s %d|%d|%d seed %d" (D.name dt) c.before c.count c.after
    c.seed

(* The positions in [D.all] of the complex dtypes, 2 of 21: drawn a quarter of
   the time besides, so that every run of 100 cases copies one (a miss once in
   about 10^17 runs, against 1 in 20000 drawn evenly). *)
let complexes =
  List.filter (fun (D.Any dt) -> D.is D.Complex dt) D.all
  |> List.map (fun d -> Option.get (List.find_index (( = ) d) D.all))

let around =
  let open Gen in
  with_pp pp_around
    (let+ dt =
       frequency
         [ (3, int_range 0 (List.length D.all - 1)); (1, of_list complexes) ]
     and+ before = int_range 0 9
     and+ count = int_range 0 600
     and+ after = int_range 0 9
     and+ seed = nat in
     { dt; before; count; after; seed })

(* A copy into elements [before, before + count) of an array keeps every other
   element's bits, those that share a byte with the destination's first or last
   included: a sub-byte dtype is copied as a loop. *)
let law_around c =
  let (D.Any dt) = List.nth D.all c.dt in
  cover "sub-byte" (D.bits dt < 8);
  cover "complex" (D.is D.Complex dt);
  cover "several groups" (c.count > 256);
  cover "odd start" (c.before mod 2 = 1);
  let st = Random.State.make [| c.seed |] in
  let n = c.before + c.count + c.after in
  let whole = drawn st dt [| n |] in
  let x = drawn st dt [| c.count |] in
  let range = { M.start = c.before; count = c.count; step = 1 } in
  let dst = Option.get (A.move (M.Slice [| range |]) whole) in
  let was = elements whole in
  let got = Support.map1 (module Nx_metal) P.Copy ~dst x in
  equal answer A.Done got;
  let want =
    Array.concat
      [
        Array.sub was 0 c.before;
        elements x;
        Array.sub was (c.before + c.count) c.after;
      ]
  in
  equal (array string) want (elements whole)

(* Outputs that read each other's destinations *)

let rotatable = List.filter (fun (D.Any dt) -> D.bits dt >= 8) D.all

type rotation = {
  dt : int;
  outputs : int;
  shape : int array;
  fresh : bool;
  budget : float option;
  seed : int;
}

let pp_rotation ppf c =
  let (D.Any dt) = List.nth rotatable c.dt in
  Format.fprintf ppf "%s, %d outputs of [%s]%s%s, seed %d" (D.name dt) c.outputs
    (String.concat "; " (Array.to_list (Array.map string_of_int c.shape)))
    (if c.fresh then ", workspace of its own" else "")
    (match c.budget with
    | None -> ""
    | Some t -> Printf.sprintf ", launches of %g s" t)
    c.seed

(* Budgets of launches that copy from tens to thousands of contiguous elements,
   past what one group of the copy takes: past a launch's dispatch by 0.3 to 1.5
   us. *)
let small_budget =
  Gen.float_range (Plan.dispatch +. 3e-7) (Plan.dispatch +. 1.5e-6)

let rotation =
  let open Gen in
  let shape =
    let* r = int_range 0 3 in
    array ~size:(constant r) (frequency [ (1, constant 0); (6, int_range 1 7) ])
  in
  let long = map (fun n -> [| n |]) (int_range 257 3000) in
  with_pp pp_rotation
    (let+ dt = int_range 0 (List.length rotatable - 1)
     and+ outputs = int_range 1 4
     and+ shape = frequency [ (3, shape); (1, long) ]
     and+ fresh = bool
     and+ budget = option small_budget
     and+ seed = nat in
     { dt; outputs; shape; fresh; budget; seed })

(* Rotations whose outputs each take several copy launches, of elements of 1, 4
   and 16 bytes at the least budget: the generator draws one in a few cases of a
   hundred, and some seeds none. *)
let rotation_examples =
  let dt name =
    let rec find i = function
      | D.Any d :: _ when D.name d = name -> i
      | _ :: l -> find (i + 1) l
      | [] -> invalid_arg name
    in
    find 0 rotatable
  in
  let budget = Some (Plan.dispatch +. 3e-7) in
  [
    {
      dt = dt "int8";
      outputs = 1;
      shape = [| 3000 |];
      fresh = false;
      budget;
      seed = 1;
    };
    {
      dt = dt "float32";
      outputs = 2;
      shape = [| 3000 |];
      fresh = true;
      budget;
      seed = 2;
    };
    {
      dt = dt "complex128";
      outputs = 4;
      shape = [| 257 |];
      fresh = false;
      budget;
      seed = 3;
    };
  ]

let kept_workspace = 64 * 1024 * 1024
let kept_tickets = 4 * 1024 * 1024
let kept_plans = 2 * 1024 * 1024

(* Runs [f] with every call taking workspace and tickets of its own. *)
let all_fresh f =
  let restore () =
    Nx_metal.bound_fresh ~workspace:kept_workspace ~tickets:kept_tickets
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_metal.bound_fresh ~workspace:0 ~tickets:0;
      f ())

(* Each of [got]'s destinations holds its partner's elements in [want], each the
   same bits or both NaN: compared whole, element by element only where the
   bytes differ. *)
let same_bits want got =
  Elements.matching (fun d -> Testable.equal (Elements.same d)) want got

(* Runs [f] with launches of at most [s] seconds. *)
let bounded s f =
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget s;
      f ())

(* The copy launches of a map of [k] outputs that each read another's
   destination, in one launch each: a staging copy and a copy an output, or one
   copy for a lone output, which reads its own. *)
let unsplit k = if k = 1 then 1 else 2 * k

(* Whether [k] copies elements one at a time: copy, or the copy of a unit of
   32-bit indices. *)
let element_copy (k : Nx_metal.Kernels.t) =
  match k with
  | Copy | Copy_8 | Copy_16 | Copy_32 | Copy_64 -> true
  | _ -> false

(* Whether [k] copies a source whose elements are apart: an element at a time,
   or rows of 16 bytes or more of consecutive elements or of every other one. *)
let apart_copy (k : Nx_metal.Kernels.t) =
  match k with
  | Copy_rows | Copy_rows_wide | Copy_step_8 | Copy_step_16 | Copy_step_32
  | Copy_step_64 ->
      true
  | _ -> element_copy k

let copies launches =
  List.length
    (List.filter
       (fun (l : Plan.traced) ->
         apart_copy l.kernel || l.kernel = Nx_metal.Kernels.Copy_run)
       launches)

(* A map whose output [k] is operand [k + 1] (modulo the outputs), each
   destination the operand of the same number, gives each destination the bits
   its next operand held when the call began: a destination may be an operand
   read at its own index, and another output reads it at its own index too. *)
let law_rotation c =
  let (D.Any dt) = List.nth rotatable c.dt in
  let k = c.outputs in
  cover "one output" (k = 1);
  cover "several outputs" (k > 1);
  cover "no element" (Array.exists (( = ) 0) c.shape);
  cover "workspace of its own" c.fresh;
  let st = Random.State.make [| c.seed |] in
  let xs = Array.init k (fun _ -> drawn st dt c.shape) in
  let was = Array.map elements xs in
  let p =
    P.v
      ~ins:(Iarray.init k (fun _ -> D.Any dt))
      (Iarray.init k (fun i -> P.In i))
      ~outs:(Iarray.init k (fun i -> (i + 1) mod k))
  in
  let dsts = Iarray.map (fun x -> A.Any x) (Iarray.of_array xs) in
  let loads = Iarray.map (fun x -> Nx_kernel.Plain x) (Iarray.of_array xs) in
  let call () = Nx_metal.map p ~dsts loads in
  let call () =
    match c.budget with None -> call () | Some t -> bounded t call
  in
  let call () =
    Plan.traced (fun () -> if c.fresh then all_fresh call else call ())
  in
  let got, launches = call () in
  let n = Array.fold_left ( * ) 1 c.shape in
  cover "an output in several launches" (n > 0 && copies launches > unsplit k);
  equal answer A.Done got;
  equal
    (array (array string))
    (Array.init k (fun i -> was.((i + 1) mod k)))
    (Array.map elements xs)

(* Copies of any source *)

let widths = [| 1; 2; 4; 8; 16 |]
let of_width w = List.filter (fun (D.Any dt) -> D.bits dt = 8 * w) rotatable

type source = {
  width : int;
  dt : int;
  src_at : int;
  dst_at : int;
  src_base : int;
  dst_base : int;
  rows : int;
  cols : int;
  step : int;
  gap : int;
  sbudget : float option;
  sseed : int;
}

let source_dtype c = List.nth (of_width widths.(c.width)) c.dt

(* The source's layout: [rows] of [cols] elements [step] apart, a row [gap]
   elements past the previous one's end, from position [src_at]. *)
let source_layout c =
  L.v ~offset:c.src_at
    ~strides:[| (c.cols * c.step) + c.gap; c.step |]
    [| c.rows; c.cols |]

let pp_source ppf c =
  let (D.Any dt) = source_dtype c in
  Format.fprintf ppf "%s %a at byte %d into %d at byte %d%s, seed %d"
    (D.name dt) L.pp (source_layout c) c.src_base c.dst_at c.dst_base
    (match c.sbudget with
    | None -> ""
    | Some t -> Printf.sprintf ", launches of %g s" t)
    c.sseed

(* Budgets of launches that copy tens to thousands of elements apart, whose axes
   past the first weigh a 64-bit division each. *)
let apart_budget = Gen.float_range 1e-5 5e-5

(* The byte a view starts at in a buffer that starts on 256 bytes: 0, or 1, 2,
   4, 8 or 12 where elements stored on [align] bytes may start. *)
let base ~align =
  let open Gen in
  let past = List.filter (fun b -> b mod align = 0) [ 1; 2; 4; 8; 12 ] in
  frequency ((3, constant 0) :: List.map (fun b -> (1, constant b)) past)

let source =
  let open Gen in
  let any = frequency [ (1, constant 0); (2, int_range 1 40) ] in
  (* Positions 16 elements apart in buffers that start on 16 bytes keep the ends
     on 16 bytes at every width: a third of the cases. *)
  let on16 =
    frequency [ (1, constant 0); (1, map (( * ) 16) (int_range 1 2)) ]
  in
  with_pp pp_source
    (let* width = int_range 0 (Array.length widths - 1) in
     let* placed = frequency [ (1, constant true); (2, constant false) ] in
     let offset = if placed then on16 else any in
     let* c =
       let+ dt = int_range 0 (List.length (of_width widths.(width)) - 1)
       and+ src_at = offset
       and+ dst_at = offset
       and+ rows = frequency [ (3, constant 1); (1, int_range 2 4) ]
       and+ cols =
         frequency
           [
             (1, constant 0);
             (1, constant 1);
             (3, int_range 2 64);
             (3, int_range 65 5000);
           ]
       and+ step = frequency [ (4, constant 1); (1, int_range 2 3) ]
       and+ gap = frequency [ (4, constant 0); (1, int_range 1 3) ] in
       {
         width;
         dt;
         src_at;
         dst_at;
         src_base = 0;
         dst_base = 0;
         rows;
         cols;
         step;
         gap;
         sbudget = None;
         sseed = 0;
       }
     in
     let budget =
       if L.is_contiguous (source_layout c) then small_budget else apart_budget
     in
     let (D.Any dt) = source_dtype c in
     let align = if D.is D.Complex dt then D.bits dt / 16 else D.bits dt / 8 in
     let base = if placed then constant 0 else base ~align in
     let+ src_base = base
     and+ dst_base = base
     and+ sbudget = option budget
     and+ sseed = nat in
     { c with src_base; dst_base; sbudget; sseed })

(* Sources whose draws are rarest, so that each run reaches their covers: a run
   of 4 bytes within a block of 16, 2 bytes past the destination's modulo 16 (3
   % of the draws); a run of 80,000 bytes, 8 past the destination's, in launches
   of the least budget (4 %); two rows of 5 elements, 2 apart (4 %). *)
let source_examples =
  let c ?sbudget ~width ~src_at ~dst_at ~src_base ~rows ~cols ~gap sseed =
    {
      width;
      dt = 0;
      src_at;
      dst_at;
      src_base;
      dst_base = 0;
      rows;
      cols;
      step = 1;
      gap;
      sbudget;
      sseed;
    }
  in
  [
    c ~width:0 ~src_at:7 ~dst_at:5 ~src_base:0 ~rows:1 ~cols:4 ~gap:0 1;
    c ~width:4 ~src_at:0 ~dst_at:0 ~src_base:8 ~rows:1 ~cols:5000 ~gap:0
      ~sbudget:(Plan.dispatch +. 3e-7) 2;
    c ~width:2 ~src_at:1 ~dst_at:0 ~src_base:0 ~rows:3 ~cols:5 ~gap:2 3;
  ]

(* The most bytes, at most 16, that the addresses [src] and [dst] of a copy's
   ends' first bytes are multiples of. *)
let placed_on ~src ~dst =
  let rec widest n =
    if n = 1 || (src mod n = 0 && dst mod n = 0) then n else widest (n / 2)
  in
  widest 16

(* [n] contiguous [dt] elements of random bits over the bytes of a buffer from
   its byte [base] on. *)
let drawn_at st dt ~base n =
  let bytes = n * (D.bits dt / 8) in
  let whole = A.buffer (drawn st D.Uint8 [| base + bytes |]) in
  let view = Rig.Buffer.view whole ~first:base ~length:bytes in
  A.v dt (L.contiguous [| n |]) view

(* nx_metal.mli: whether a copy of [l], of [width]-byte elements, copies rows:
   its innermost axis, once adjacent axes that step alike are merged, a row of
   consecutive elements of 16 bytes or more beside another axis. *)
let by_rows ~width l =
  let m = (L.coalesce [| L.contiguous (L.shape l); l |]).(1) in
  let r = L.rank m in
  r >= 2 && L.stride m (r - 1) = 1 && L.dim m (r - 1) * width >= 16

(* nx_metal.mli: the fewest bytes of a row of every other element of [width]
   bytes copy_step takes where a block of the destination crosses a row's end,
   if any. *)
let step_row_bytes = function 1 | 2 -> Some 96 | 4 -> Some 192 | _ -> None

(* Whether [row] bytes lie under {!step_row_bytes} of [width]. *)
let under_least ~width row =
  match step_row_bytes width with Some least -> row < least | None -> true

(* nx_metal.mli: whether a copy of [l], of [width]-byte elements stored on
   [align] bytes, into a destination whose first byte lies [head] bytes past 16,
   copies rows of every other element: its innermost axis, once adjacent axes
   that step alike are merged, steps two elements of 8 bytes or fewer stored on
   their width, in a copy whose byte indices take 32 bits, to [narrow], over
   {!step_row_bytes} or more of them where it has a bound, or 16 or more where
   no block of the destination crosses a row's end: one row, or rows of a
   multiple of 16 bytes from [head] 0. *)
let by_steps ~narrow ~width ~align ~head l =
  let m = (L.coalesce [| L.contiguous (L.shape l); l |]).(1) in
  let r = L.rank m in
  let row = L.dim m (r - 1) * width in
  let uncrossed = r = 1 || (row mod 16 = 0 && head = 0) in
  L.stride m (r - 1) = 2
  && width <= 8 && align = width
  && L.numel l * width <= narrow
  && row >= 16
  && (uncrossed || not (under_least ~width row))

(* The kernel of a launch that ends at byte [stop] of a copy of rows of [l], of
   [width]-byte elements: copy_rows where its byte indices and every extent of
   its merged axes, its row's in bytes, take 32 bits, to [narrow]. *)
let rows_kernel ~narrow ~width l stop =
  let m = (L.coalesce [| L.contiguous (L.shape l); l |]).(1) in
  let r = L.rank m in
  let extent j = if j = r - 1 then L.dim m j * width else L.dim m j in
  let wide = List.exists (fun j -> extent j > narrow) (List.init r Fun.id) in
  if wide || stop > narrow then "copy_rows_wide" else "copy_rows"

(* A copy of a source of any layout, from any position, into a destination at
   any position, each in a view that starts at any byte its elements may, gives
   the destination the source's elements in C order and keeps the bits around
   it: elements of every width, sources of consecutive elements whose address
   and length are no multiple of 16 bytes, and sources whose elements are apart.
   A source of consecutive elements is copied by copy_run, wherever its first
   byte and the destination's lie, a source of rows by copy_rows ([by_rows],
   [rows_kernel]), one of rows of every other element by copy_step ([by_steps]),
   and every other source by the copy of its unit where a launch ends at element
   [narrow] or before, and by copy past it: [2^32 - 256], or less under
   {!Nx_metal.bound_copy}. *)
let law_source ?(narrow = (1 lsl 32) - 256) c =
  let (D.Any dt) = source_dtype c in
  let width = widths.(c.width) in
  let st = Random.State.make [| c.sseed |] in
  let l = source_layout c in
  let n = L.numel l in
  let _, hi = L.span l in
  let x =
    A.v dt l (A.buffer (drawn_at st dt ~base:c.src_base (Int.max hi 1)))
  in
  let after = 3 in
  let whole = drawn_at st dt ~base:c.dst_base (c.dst_at + n + after) in
  let into =
    L.v ~offset:c.dst_at ~strides:[| c.cols; 1 |] [| c.rows; c.cols |]
  in
  let dst = A.v dt into (A.buffer whole) in
  let was = elements whole in
  let call () = Support.map1 (module Nx_metal) P.Copy ~dst x in
  let call () =
    match c.sbudget with None -> call () | Some t -> bounded t call
  in
  let got, launches = Plan.traced call in
  let runs =
    List.filter
      (fun (t : Plan.traced) -> t.kernel = Nx_metal.Kernels.Copy_run)
      launches
  in
  (* Both buffers are views into ones that start on 256 bytes. *)
  let on =
    placed_on
      ~src:(c.src_base + (c.src_at * width))
      ~dst:(c.dst_base + (c.dst_at * width))
  in
  let by_position = placed_on ~src:(c.src_at * width) ~dst:(c.dst_at * width) in
  let consecutive = n > 0 && L.is_contiguous l in
  let align = if D.is D.Complex dt then width / 2 else width in
  (* The destination's first byte in its block of 16 bytes, and the source's
     bytes past the destination's, modulo 16. *)
  let src_byte = c.src_base + (c.src_at * width) in
  let dst_byte = c.dst_base + (c.dst_at * width) in
  let head = dst_byte mod 16 in
  let shift = (src_byte - dst_byte + 16) mod 16 in
  let count (t : Plan.traced) =
    Int64.to_int (Bytes.get_int64_le t.words (8 * Nx_metal.Kernels.copy_count))
  in
  let kernel t =
    if consecutive then "copy_run"
    else if by_rows ~width l then rows_kernel ~narrow ~width l (count t)
    else if by_steps ~narrow ~width ~align ~head l then
      Printf.sprintf "copy_step_%d" (8 * width)
    else if count t > narrow then "copy"
    else Printf.sprintf "copy_%d" (8 * align)
  in
  let bytes = n * width in
  cover (Printf.sprintf "%d-byte elements" width) true;
  cover "a source whose elements are apart" (n > 1 && not (L.is_contiguous l));
  cover "a source of two axes, the inner of an extent no power of 2"
    (c.rows > 1 && c.gap > 0 && c.cols > 2 && c.cols land (c.cols - 1) <> 0);
  (* The narrow bound's law draws sources apart alone. *)
  if narrow = (1 lsl 32) - 256 then begin
    cover "no element" (n = 0);
    cover "one element" (n = 1);
    List.iter
      (fun u ->
        cover
          (Printf.sprintf "consecutive elements, ends on multiples of %d" u)
          (consecutive && on = u))
      [ 1; 2; 4; 8; 16 ];
    cover "a base that takes the ends off 16 bytes"
      (consecutive && by_position = 16 && on < 16);
    cover "a tail past the last 16 bytes" (runs <> [] && n * width mod 16 <> 0);
    cover "a source at the destination's byte modulo 16"
      (consecutive && shift = 0);
    cover "a source 1 to 3 bytes past the destination's modulo 16"
      (consecutive && shift >= 1 && shift <= 3);
    cover "a source 4, 8 or 12 bytes past the destination's modulo 16"
      (consecutive && shift mod 4 = 0 && shift <> 0);
    cover "a source 5 to 15 bytes past, no multiple of 4"
      (consecutive && shift > 4 && shift mod 4 <> 0);
    cover "a destination off 16 bytes" (consecutive && head <> 0);
    cover "a run within one block of 16 bytes, off both its ends"
      (consecutive && head > 0 && head + bytes < 16);
    cover "a run under 32 bytes" (consecutive && bytes < 32);
    let groups (t : Plan.traced) =
      let g, _, _ = t.groups in
      g
    in
    cover "a run of several groups" (List.exists (fun t -> groups t > 1) runs);
    cover "a run in several launches" (List.length runs > 1);
    cover "a shifted run in several launches"
      (List.length runs > 1 && shift <> 0)
  end;
  equal answer A.Done got;
  if narrow < (1 lsl 32) - 256 && not consecutive then begin
    let launches =
      List.filter (fun (t : Plan.traced) -> element_copy t.kernel) launches
    in
    let wide t = count t > narrow in
    cover "a launch of 64-bit indices" (List.exists wide launches);
    cover "a launch of 32-bit indices, the copy past them"
      (n > narrow && List.exists (fun t -> not (wide t)) launches);
    cover "a launch at element [narrow], of 32-bit indices"
      (List.exists (fun t -> count t = narrow) launches)
  end;
  let name (t : Plan.traced) = Nx_metal.Kernels.name t.kernel in
  equal (list string) (List.map kernel launches) (List.map name launches);
  let want =
    Array.concat
      [
        Array.sub was 0 c.dst_at; elements x; Array.sub was (c.dst_at + n) after;
      ]
  in
  equal (array string) want (elements whole)

(* The seconds {!Plan.seconds} gives a launch of one group of copy threads of
   64-bit indices over two axes, with 16 KiB of traffic: by nx_metal.mli, 128
   operations a thread and 4096 for its inner axis. The planner bounds a
   launch's elements more cautiously, so launches within it copy one element or
   block, or a few. *)
let one_group =
  Plan.seconds ~threads:256 ~groups:(1, 1, 1)
    { Plan.bytes = 16 * 1024; ops = 256 * (128 + 4096); chain = 0 }

(* Budgets just past the planner's estimate of a launch of 64-bit indices over
   two axes, 10.4 us: a dispatch and 4224 operations a thread, over two rounds
   of 128 lanes. Their launches copy a few to a few hundred elements, so that a
   copy past element 256 starts with launches of 32-bit indices. *)
let few_budget = Gen.float_range 1.05e-5 1.15e-5

(* Sources whose elements are apart, of 255 to 257 elements in a row, or of 65
   to 8000: around and past the 256 elements of 32-bit indices [law_source_wide]
   bounds launches to, under the budgets of sources whose elements are apart:
   [source]'s, drawn for its own layout, may be a contiguous source's, too small
   for one element apart. *)
let source_wide =
  let open Gen in
  with_pp pp_source
    (let* c = source in
     let* step = int_range 2 3 in
     let* near = frequency [ (1, constant true); (2, constant false) ] in
     let* rows = if near then constant 1 else int_range 1 4 in
     let* sbudget = option (frequency [ (1, few_budget); (2, apart_budget) ]) in
     let+ cols =
       if near then of_list [ 255; 256; 257 ] else int_range 65 2000
     in
     { c with rows; cols; step; sbudget })

(* A copy past element 256 in launches of one or a few elements, whose first
   ones take 32-bit indices and end at element 256, whatever the margin of
   {!few_budget}'s draws over the planner's estimate. *)
let wide_examples =
  [
    {
      width = 2;
      dt = 0;
      src_at = 0;
      dst_at = 0;
      src_base = 0;
      dst_base = 0;
      rows = 2;
      cols = 150;
      step = 3;
      gap = 1;
      sbudget = Some one_group;
      sseed = 1;
    };
  ]

(* [law_source] where a launch's indices take 64 bits past its element 256. *)
let law_source_wide c =
  Fun.protect
    ~finally:(fun () -> Nx_metal.bound_copy ~wide:32)
    (fun () ->
      Nx_metal.bound_copy ~wide:9;
      law_source ~narrow:256 c)

(* A source of rows: [outer] blocks of [nrows] rows of [cols] elements of
   [widths.(rw)] bytes, a row [gap] elements past the previous one's end and a
   block [gap] past the previous block's, the rows in reverse where [flip],
   within [deep] axes more of extent 2 that merge with none; the elements of a
   row [rstep] apart; [at] elements past its buffer's first, the buffer ending
   at its last byte. *)
type rows = {
  rw : int;
  rdt : int;
  deep : int;
  outer : int;
  nrows : int;
  cols : int;
  rstep : int;
  gap : int;
  flip : bool;
  at : int;
  rbase : int;
  dbase : int;
  dat : int;
  rbudget : float option;
  rseed : int;
}

let rows_dtype c = List.nth (of_width widths.(c.rw)) c.rdt

let rows_layout c =
  let rs = (c.cols * c.rstep) + c.gap in
  let os = (c.nrows * rs) + c.gap in
  let rs' = if c.flip then -rs else rs in
  let offset = c.at + if c.flip then (c.nrows - 1) * rs else 0 in
  let shape, strides =
    if c.outer > 1 then ([ c.outer; c.nrows; c.cols ], [ os; rs'; c.rstep ])
    else ([ c.nrows; c.cols ], [ rs'; c.rstep ])
  in
  (* Each axis more steps past the span of those inside it and a gap, so that it
     merges with none. *)
  let rec deepen k shape strides =
    if k = 0 then (shape, strides)
    else
      let span =
        List.fold_left2 (fun a d t -> a + ((d - 1) * abs t)) 1 shape strides
      in
      let t = span + c.gap + 1 in
      let t = if t = List.hd shape * List.hd strides then t + 1 else t in
      deepen (k - 1) (2 :: shape) (t :: strides)
  in
  let shape, strides = deepen c.deep shape strides in
  L.v ~offset ~strides:(Iarray.of_list strides) (Iarray.of_list shape)

let pp_rows ppf c =
  let (D.Any dt) = rows_dtype c in
  Format.fprintf ppf "%s %a at byte %d into %d at byte %d%s, seed %d"
    (D.name dt) L.pp (rows_layout c) c.rbase c.dat c.dbase
    (match c.rbudget with
    | None -> ""
    | Some t -> Printf.sprintf ", launches of %g s" t)
    c.rseed

(* Budgets of launches that copy rows: a launch's fixed cost and work for one
   block, 0.5 us on the M1 Max at 256 operations a thread, and room for tens to
   a few thousand blocks. *)
let rows_budget =
  Gen.float_range (Plan.dispatch +. 8e-7) (Plan.dispatch +. 1.5e-6)

(* Rows of 16, 17 or 210 bytes, or any from 16 to 300, rounded up to elements,
   of 2 elements or more: a row of one element is no row. Rows within axes more
   take the whole budget, as their work grows with their axes. *)
let rows_source =
  let open Gen in
  with_pp pp_rows
    (let* rw = int_range 0 (Array.length widths - 1) in
     let w = widths.(rw) in
     let* rdt = int_range 0 (List.length (of_width w) - 1) in
     let (D.Any d) = List.nth (of_width w) rdt in
     let align = if D.is D.Complex d then w / 2 else w in
     let* bytes =
       frequency
         [
           (1, constant 16);
           (1, constant 17);
           (1, constant 210);
           (3, int_range 16 300);
         ]
     in
     let* flip = frequency [ (3, constant false); (1, constant true) ] in
     let+ deep = frequency [ (9, constant 0); (1, int_range 1 6) ]
     and+ outer = frequency [ (3, constant 1); (1, int_range 2 3) ]
     and+ nrows = int_range 2 40
     and+ gap = if flip then int_range 0 20 else int_range 1 20
     and+ at = int_range 0 3
     and+ rbase = base ~align
     and+ dbase = base ~align
     and+ dat = int_range 0 20
     and+ rbudget = option rows_budget
     and+ rseed = nat in
     let rbudget = if deep > 0 then None else rbudget in
     {
       rw;
       rdt;
       deep;
       outer;
       nrows;
       cols = Int.max 2 ((bytes + w - 1) / w);
       rstep = 1;
       gap;
       flip;
       at;
       rbase;
       dbase;
       dat;
       rbudget;
       rseed;
     })

(* [rows] under the bound of {!law_rows_wide}, with budgets past the planner's
   estimate of a launch of 64-bit byte indices over three axes, 18.4 us, but for
   rows within axes more. *)
let budgets_wide rows =
  let open Gen in
  with_pp pp_rows
    (let* c = rows in
     let+ rbudget = option (float_range 2e-5 5e-5) in
     { c with rbudget = (if c.deep > 0 then None else rbudget) })

let rows_wide = budgets_wide rows_source

(* Rows the draws reach a few times in 300: 32 bytes, under the bound of
   {!law_rows_wide}; rows of rows whose blocks of the destination cross from one
   block of rows into the next; the most axes the launch's parameters hold, 7,
   and one more, which a constant holds; and rows of 272 bytes, past that bound,
   in launches of a block each, which end under it. *)
let rows_examples =
  let c =
    {
      rw = 0;
      rdt = 0;
      deep = 0;
      outer = 1;
      nrows = 2;
      cols = 16;
      rstep = 1;
      gap = 1;
      flip = false;
      at = 0;
      rbase = 0;
      dbase = 0;
      dat = 0;
      rbudget = None;
      rseed = 1;
    }
  in
  let held = Nx_metal.Kernels.copy_rows_holds in
  [
    c;
    { c with outer = 2; nrows = 3; cols = 17 };
    { c with deep = held - 3; outer = 2; cols = 17 };
    { c with deep = held - 2; outer = 2; cols = 17; flip = true };
    { c with cols = 272; rbudget = Some one_group };
  ]

(* nx_metal.mli: a copy of rows of 16 bytes or more runs copy_rows, or copy_step
   for rows of every other element ([rstep] 2), and gives the destination the
   source's elements in C order, keeping the bits around it: rows at every shift
   from 16 bytes, blocks of the destination across two rows, rows of rows, rows
   in reverse, and sources whose span starts or ends off 16 bytes, at its
   buffer's ends. Rows of every other element under 16 bytes, or under
   {!step_row_bytes} where a block of the destination crosses a row's end, of
   complex elements, or of elements of 16 bytes are copied an element at a
   time. *)
let law_rows ?(narrow = (1 lsl 32) - 256) c =
  let (D.Any dt) = rows_dtype c in
  let width = widths.(c.rw) in
  let st = Random.State.make [| c.rseed |] in
  let l = rows_layout c in
  let n = L.numel l in
  let lo, hi = L.span l in
  let x = A.v dt l (A.buffer (drawn_at st dt ~base:c.rbase hi)) in
  let after = 3 in
  let whole = drawn_at st dt ~base:c.dbase (c.dat + n + after) in
  let shape = L.shape l in
  let into =
    L.v ~offset:c.dat ~strides:(L.strides (L.contiguous shape)) shape
  in
  let dst = A.v dt into (A.buffer whole) in
  let was = elements whole in
  let call () = Support.map1 (module Nx_metal) P.Copy ~dst x in
  let call () =
    match c.rbudget with None -> call () | Some t -> bounded t call
  in
  let got, launches = Plan.traced call in
  equal answer A.Done got;
  let name (t : Plan.traced) = Nx_metal.Kernels.name t.kernel in
  let word (t : Plan.traced) i =
    Int64.to_int (Bytes.get_int64_le t.words (8 * i))
  in
  let stop t = word t Nx_metal.Kernels.copy_rows_end in
  let align = if D.is D.Complex dt then width / 2 else width in
  let head = (c.dbase + (c.dat * width)) mod 16 in
  let steps = c.rstep = 2 && by_steps ~narrow ~width ~align ~head l in
  let kernel t =
    if c.rstep = 1 then rows_kernel ~narrow ~width l (stop t)
    else if steps then Printf.sprintf "copy_step_%d" (8 * width)
    else if word t Nx_metal.Kernels.copy_count > narrow then "copy"
    else Printf.sprintf "copy_%d" (8 * align)
  in
  equal (list string) (List.map kernel launches) (List.map name launches);
  let want =
    Array.concat
      [ Array.sub was 0 c.dat; elements x; Array.sub was (c.dat + n) after ]
  in
  equal (array string) want (elements whole);
  (* Each row's first byte modulo 16, in a buffer that starts on 256. *)
  let row_bytes = c.cols * width in
  let r = L.rank l in
  let row_at k =
    let p = ref (L.offset l) and rest = ref k in
    for j = r - 2 downto 0 do
      p := !p + (!rest mod L.dim l j * L.stride l j);
      rest := !rest / L.dim l j
    done;
    !p
  in
  List.iter
    (fun w -> cover (Printf.sprintf "%d-byte elements" w) (width = w))
    (Array.to_list widths);
  let merged = L.rank (L.coalesce [| L.contiguous shape; l |]).(1) in
  if c.rstep = 1 then begin
    let shifts =
      List.init (n / c.cols) (fun k -> (c.rbase + (row_at k * width)) mod 16)
    in
    List.iter
      (fun b -> cover (Printf.sprintf "rows of %d bytes" b) (row_bytes = b))
      [ 16; 17; 210 ];
    for s = 0 to 15 do
      cover (Printf.sprintf "a row %d bytes past 16" s) (List.mem s shifts)
    done;
    cover "a block of the destination across two rows"
      ((head + row_bytes) mod 16 <> 0 || row_bytes mod 16 <> 0);
    (* A block whose first row is the last of a block of rows reads the next
       row's place by a carry into the outer axis. *)
    cover "a block of the destination across two blocks of rows"
      (c.outer > 1
      && List.exists
           (fun j -> (head + (j * c.nrows * row_bytes)) mod 16 <> 0)
           (List.init (c.outer - 1) (fun j -> j + 1)))
  end
  else begin
    (* The destination's blocks of 16 bytes that hold only elements, by their
       first element: where it lies modulo 16 in the source, and whether the
       block's elements cross a row's end. *)
    let per = 16 / Int.min width 16 in
    let blocks =
      List.filter
        (fun k -> (c.dbase + ((c.dat + k) * width)) mod 16 = 0 && k + per <= n)
        (List.init n Fun.id)
    in
    let shift k =
      (c.rbase + ((row_at (k / c.cols) + (c.rstep * (k mod c.cols))) * width))
      mod 16
    in
    let within k = (k mod c.cols) + per <= c.cols in
    let block p = steps && List.exists p blocks in
    let under = under_least ~width row_bytes in
    cover "every other element, rows under 16 bytes" (row_bytes < 16);
    cover "every other element, rows that cross under the width's least"
      ((not steps) && row_bytes >= 16 && under
      && (head + row_bytes) mod 16 <> 0
      && merged > 1);
    cover "every other element, rows of the width's least, by copy_step"
      (steps && step_row_bytes width = Some row_bytes);
    cover
      "every other element, rows of a multiple of 16 bytes under the width's \
       least, by copy_step"
      (steps && under && merged > 1);
    List.iter
      (fun w ->
        cover
          (Printf.sprintf "every other element of %d bytes, by copy_step" w)
          (steps && width = w))
      [ 1; 2; 4; 8 ];
    cover "every other complex element" (D.is D.Complex dt && width <= 8);
    cover "every other element of one row" (steps && merged = 1);
    cover "a block from two aligned words on 16 bytes"
      (block (fun k -> within k && shift k = 0));
    cover "a block from two aligned words off 16 bytes"
      (block (fun k -> within k && shift k > 0 && shift k <= width));
    cover "a block from three aligned words"
      (block (fun k -> within k && shift k > width));
    cover "a block of the destination across two rows"
      (block (fun k -> not (within k)))
  end;
  cover "rows of rows" (c.outer > 1);
  cover "rows of the most axes the launch holds"
    (merged = Nx_metal.Kernels.copy_rows_holds);
  cover "rows of more axes than the launch holds, from a constant"
    (merged > Nx_metal.Kernels.copy_rows_holds);
  cover "rows in reverse" c.flip;
  cover "rows of rows in reverse" (c.outer > 1 && c.flip);
  cover "a source at its buffer's first byte, off 16 bytes"
    (lo = 0 && c.rbase mod 16 <> 0);
  cover "a source to its buffer's last byte, off 16 bytes"
    ((c.rbase + (hi * width)) mod 16 <> 0);
  cover "a copy in several launches" (List.length launches > 1);
  if narrow < (1 lsl 32) - 256 && c.rstep = 2 then begin
    cover "every other element past the bound, an element a thread"
      ((not steps) && by_steps ~narrow:max_int ~width ~align ~head l);
    cover "every other element under the bound, by copy_step" steps
  end;
  if narrow < (1 lsl 32) - 256 && c.rstep = 1 then begin
    let wide t = stop t > narrow in
    cover "a launch of 64-bit byte indices" (List.exists wide launches);
    cover "a launch of 32-bit byte indices"
      (List.exists (fun t -> not (wide t)) launches);
    cover "a row past the bound in bytes, in a launch that ends under it"
      (row_bytes > narrow && List.exists (fun t -> not (wide t)) launches)
  end

(* [law_rows] where a launch's byte indices take 64 bits past byte 256. *)
let law_rows_wide c =
  Fun.protect
    ~finally:(fun () -> Nx_metal.bound_copy ~wide:32)
    (fun () ->
      Nx_metal.bound_copy ~wide:9;
      law_rows ~narrow:256 c)

(* Rows of every other element: under 16 bytes; of one element under, at or over
   the least bytes copy_step takes a row of the width at, where it has one; of a
   multiple of 16 bytes up to it, or to 600 bytes where it has none; or any up
   to 600 bytes; of 2 elements or more; single rows, and rows no gap apart,
   which merge into one. *)
let steps_source =
  let open Gen in
  with_pp pp_rows
    (let* c = rows_source in
     let w = widths.(c.rw) in
     let sixteens = Int.max 1 (16 / w) in
     let short = int_range 2 (Int.max 2 (15 / w)) in
     let any = int_range 2 (Int.max 2 (600 / w)) in
     let to_bytes b = map (fun k -> k * sixteens) (int_range 1 (b / 16)) in
     let* cols =
       match step_row_bytes w with
       | None -> frequency [ (1, short); (1, to_bytes 600); (3, any) ]
       | Some least ->
           let near = least / w in
           frequency
             [
               (1, short);
               (2, int_range (near - 1) (near + 1));
               (1, to_bytes least);
               (3, any);
             ]
     in
     let* nrows = frequency [ (1, constant 1); (3, int_range 2 40) ] in
     let+ gap =
       if c.flip then int_range 0 20
       else frequency [ (1, constant 0); (3, int_range 1 20) ]
     in
     { c with rstep = 2; cols = Int.max 2 cols; nrows; gap })

let steps_wide = budgets_wide steps_source

(* Rows of every other element of the most axes the launch's parameters hold, 7,
   and of one more; and, under the bound of {!law_rows_wide}, whose draws reach
   few, two rows into a destination on 16 bytes: at each width copy_step takes,
   of one element under, at and over its least bytes, or of 72 and 264 bytes
   where it has none, and of 32 bytes; of uint8 rows of 9 bytes; and of complex
   elements. *)
let steps_examples =
  let c = { (List.hd rows_examples) with rstep = 2 } in
  let complex =
    let rec find i = function
      | [] -> invalid_arg "steps_examples: no complex dtype of 8 bytes"
      | D.Any dt :: rest -> if D.is D.Complex dt then i else find (i + 1) rest
    in
    find 0 (of_width 8)
  in
  List.map
    (fun c -> { c with rstep = 2; cols = 65 })
    [ List.nth rows_examples 2; List.nth rows_examples 3 ]
  @ List.concat_map
      (fun rw ->
        let w = widths.(rw) in
        let near =
          match step_row_bytes w with
          | Some least -> [ (least / w) - 1; least / w; (least / w) + 1 ]
          | None -> [ 72 / w; 264 / w ]
        in
        List.map (fun cols -> { c with rw; cols }) (near @ [ 32 / w ]))
      [ 0; 1; 2; 3 ]
  @ [ { c with rw = 0; cols = 9 }; { c with rw = 3; rdt = complex; cols = 9 } ]

(* Each operand another output's destination shares memory with is first copied
   into workspace: two launches for it, one for an output whose operand is its
   own destination or none's. *)
let test_staged () =
  let st = Random.State.make [| 3 |] in
  let xs = Array.init 2 (fun _ -> drawn st D.Int32 [| 1000 |]) in
  let y = drawn st D.Int32 [| 1000 |] in
  let fresh = A.create (S.device ()) D.Int32 [| 1000 |] in
  let ins = Iarray.init 3 (fun _ -> D.Any D.Int32) in
  let p = P.v ~ins (Iarray.init 3 (fun i -> P.In i)) ~outs:[| 1; 0; 2 |] in
  let dsts : A.any iarray = [| A.Any xs.(0); A.Any xs.(1); A.Any fresh |] in
  let loads : Nx_kernel.load iarray =
    [| Plain xs.(0); Plain xs.(1); Plain y |]
  in
  let want = [| elements xs.(1); elements xs.(0); elements y |] in
  let got, launches = Plan.traced (fun () -> Nx_metal.map p ~dsts loads) in
  equal answer A.Done got;
  equal int 5 (List.length launches);
  equal
    (array (array string))
    want
    [| elements xs.(0); elements xs.(1); elements fresh |]

(* nx_metal.mli: a staged operand takes the elements its layout reaches where
   they are fewer than its own. A row broadcast to 1000 rows, past another
   output's destination in that destination's buffer, takes a row of workspace,
   and both outputs get the bits the call found. *)
let test_staged_broadcast () =
  let st = Random.State.make [| 4 |] in
  let x = drawn st D.Int32 [| 1000; 3 |] in
  let b = A.buffer (drawn st D.Int32 [| 3003 |]) in
  let y = A.v D.Int32 (L.contiguous [| 1000; 3 |]) b in
  let rows = L.v ~offset:3000 ~strides:[| 0; 1 |] [| 1000; 3 |] in
  let row = A.v D.Int32 rows b in
  let other = drawn st D.Int32 [| 1000; 3 |] in
  let ins : D.any iarray = [| D.Any D.Int32; D.Any D.Int32 |] in
  let p = P.v ~ins [| P.In 0; P.In 1 |] ~outs:[| 0; 1 |] in
  let dsts : A.any iarray = [| A.Any x; A.Any y |] in
  let loads : Nx_kernel.load iarray = [| Plain row; Plain other |] in
  let want = (elements row, elements other) in
  (match moves_map p ~dsts loads with
  | Ok (Some c) -> equal ~msg:"workspace" int (3 * 4) c.workspace
  | Ok None | Error _ -> fail "the moves planned no copy");
  equal answer A.Done (Nx_metal.map p ~dsts loads);
  equal (pair (array string) (array string)) want (elements x, elements y)

(* nx_metal.mli: a call's own workspace returns once its work is done, so swaps
   in a loop hold at most one call's: two operands' bytes. *)
let test_fresh_held () =
  let b = backend () in
  let n = 70_000 in
  let st = Random.State.make [| n |] in
  let x = drawn st D.Int64 [| n |] and y = drawn st D.Int64 [| n |] in
  let ins : D.any iarray = [| D.Any D.Int64; D.Any D.Int64 |] in
  let p = P.v ~ins [| P.In 0; P.In 1 |] ~outs:[| 1; 0 |] in
  let dsts : A.any iarray = [| A.Any x; A.Any y |] in
  let loads : Nx_kernel.load iarray = [| Plain x; Plain y |] in
  let piece = ((8 * n) + 255) land lnot 255 in
  all_fresh (fun () ->
      Kernels_cases.loop_held ~calls:8 b ~bound:(2 * piece) (fun () ->
          equal answer A.Done (Nx_metal.map p ~dsts loads);
          Rig.Buffer.wait (A.buffer x) Read_write))

(* Tickets through the door *)

(* The copy of the first [n] of [held] tickets into the first [n] words of the
   call's operand 0, a launch of the copy kernel with its axes, one of [n]
   elements, in its words. *)
let tickets_into b ~n ~held =
  let module K = Nx_metal.Kernels in
  let at = Plan.tickets b held in
  let w = Plan.words K.Copy in
  Plan.set w K.copy_src at;
  Plan.set w K.copy_count n;
  Plan.set w K.copy_rank 1;
  Plan.set w K.copy_unit 4;
  Plan.set w K.copy_units 1;
  Plan.set w (K.copy_held + K.axis_extent) n;
  Plan.set w (K.copy_held + K.axis_stride) 1;
  Plan.launch b K.Copy
    ~groups:((n + 255) / 256, 1, 1)
    ~cost:
      (cost
         ~bytes:((8 * n) + (4 * Plan.line))
         ~ops:(128 * 256 * ((n + 255) / 256))
         ())
    w
    [|
      { Plan.word = K.copy_dst; slot = Plan.slot b 0; write = true };
      { Plan.word = K.copy_src; slot = Plan.tickets_slot b; write = false };
    |]

(* plan.mli: a call's tickets are zero at its start, the device's or of its own.
   A call that copies its tickets into an array of drawn bits zeroes it, after a
   staged rotation has left bits in the workspace, so tickets bound to the
   workspace or an array would show. Tickets of a call's own are zero without
   the launches that zero them, since Metal gives new shared buffers zeroed, so
   no test shows those launches missing. Under a budget of a few words a launch,
   the zeroing takes several launches, each within the budget. *)
let test_tickets_door () =
  let d = S.device () in
  let st = Random.State.make [| 13 |] in
  let n = 1000 in
  let staged () =
    let xs = Array.init 2 (fun _ -> drawn st D.Int32 [| n |]) in
    let ins : D.any iarray = [| D.Any D.Int32; D.Any D.Int32 |] in
    let p = P.v ~ins [| P.In 0; P.In 1 |] ~outs:[| 1; 0 |] in
    let dsts = Iarray.map (fun x -> A.Any x) (Iarray.of_array xs) in
    let loads = Iarray.map (fun x -> Nx_kernel.Plain x) (Iarray.of_array xs) in
    equal answer A.Done (Nx_metal.map p ~dsts loads)
  in
  let call ?(split = false) () =
    let x = drawn st D.Int32 [| n |] in
    let b = Plan.make () in
    tickets_into b ~n ~held:8000;
    let c = Plan.finish b in
    if split then begin
      at_least ~msg:"zeroing launches" int ~than:2 c.zeroing;
      Array.iteri
        (fun i (l : Plan.launch) ->
          if i < c.zeroing then
            at_most ~msg:"a zeroing launch" float_exact ~than:(Plan.allowed ())
              (Plan.seconds
                 ~threads:(Nx_metal.Kernels.threads l.kernel)
                 ~groups:l.groups ~resident:l.resident l.cost))
        c.launches
    end;
    equal answer A.Done (Nx_metal.call d ~written:[| A.Any x |] ~read:[||] c);
    equal (array string) (Array.make n "\000\000\000\000") (elements x)
  in
  staged ();
  call ();
  call ();
  all_fresh call;
  all_fresh (fun () -> bounded (Plan.dispatch +. 5e-7) (call ~split:true))

(* Kept constants and submissions *)

(* A view of [rank] axes of 2 that no merge shortens, the outer ones 3^12, 3^11,
   ... elements apart and the innermost [steps]: for [steps] above 1, a layout
   no other test makes. *)
let deep st ~rank steps =
  let rec pow3 n = if n = 0 then 1 else 3 * pow3 (n - 1) in
  let strides =
    Array.init rank (fun j ->
        if j = rank - 1 then steps else pow3 (rank - 1 - j))
  in
  let shape = Array.make rank 2 in
  let l = L.v ~strides:(Iarray.of_array strides) (Iarray.of_array shape) in
  let _, hi = L.span l in
  A.v D.Int8 l (A.buffer (drawn st D.Int8 [| hi |]))

let copied x =
  let dst = A.create (S.device ()) D.Int8 (L.shape (A.layout x)) in
  equal answer A.Done (Support.map1 (module Nx_metal) P.Copy ~dst x);
  equal (array string) (elements x) (elements dst)

(* The constants of a plan's slots, by their bytes. *)
let constants x =
  let dst = A.create Rig.host D.Int8 (L.shape (A.layout x)) in
  let p = P.v ~ins:[| D.Any D.Int8 |] [| P.In 0 |] ~outs:[| 0 |] in
  match moves_map p ~dsts:[| A.Any dst |] [| Nx_kernel.Plain x |] with
  | Ok (Some c) ->
      Array.to_list c.slots
      |> List.filter_map (function
        | Plan.Constant { length; _ } -> Some length
        | Operand _ | Workspace | Tickets | Fresh _ | Uploaded _ -> None)
  | Ok None | Error _ -> failwith "the moves planned no copy"

(* nx_metal.mli: a copy takes a constant of 24 bytes an axis for an operand of
   more than 8 axes once merged, and none otherwise; both copy alike. *)
let test_holds () =
  let st = Random.State.make [| 17 |] in
  let held = deep st ~rank:8 2 and tail = deep st ~rank:9 2 in
  equal (list int) [] (constants held);
  equal (list int) [ 9 * 24 ] (constants tail);
  copied held;
  copied tail

(* nx_metal.mli: a device keeps at most 16 constants, and a call whose constant
   it does not keep computes alike. *)
let test_constants () =
  let d = S.device () in
  let st = Random.State.make [| 11 |] in
  let before = Nx_metal.kept_constants d in
  Fun.protect
    ~finally:(fun () -> Nx_metal.bound_constants (64 * 1024 * 1024))
    (fun () ->
      Nx_metal.bound_constants 0;
      copied (deep st ~rank:13 3));
  equal ~msg:"past the bound" int before (Nx_metal.kept_constants d);
  for steps = 4 to 24 do
    copied (deep st ~rank:13 steps)
  done;
  equal ~msg:"within it" int 16 (Nx_metal.kept_constants d)

(* plan.mli finish: each part is kept as a call of one part is, so parts of one
   key share one kept submission. Five copies into fresh arrays at a launch a
   part are five parts of one key: the call keeps at most one submission, none
   where another call kept that key first. *)
let test_parts_share () =
  let d = S.device () in
  let st = Random.State.make [| 11 |] in
  let x = drawn st D.Int8 [| 1000 |] in
  let ins : D.any iarray = [| D.Any D.Int8 |] in
  let p = P.v ~ins [| P.In 0 |] ~outs:[| 0; 0; 0; 0; 0 |] in
  let ys = Array.init 5 (fun _ -> A.create d D.Int8 [| 1000 |]) in
  let dsts = Iarray.map (fun y -> A.Any y) (Iarray.of_array ys) in
  let before = Nx_metal.kept_calls d in
  let (), launches =
    S.parted ~launches:1 (fun () ->
        Plan.traced (fun () ->
            equal answer A.Done (Nx_metal.map p ~dsts [| Nx_kernel.Plain x |])))
  in
  equal ~msg:"parts" int 5 (List.length launches);
  Array.iter (fun y -> equal (array string) (elements x) (elements y)) ys;
  at_most ~msg:"kept" int ~than:(before + 1) (Nx_metal.kept_calls d)

(* nx_metal.mli: a call past the launches kept takes a submission of its own and
   leaves the kept ones as they were; the same call, kept, adds its own once.
   Five copies into fresh arrays in one call have a key no other test makes. *)
let test_unkept () =
  let d = S.device () in
  let st = Random.State.make [| 7 |] in
  let x = drawn st D.Int8 [| 1000 |] in
  let ins : D.any iarray = [| D.Any D.Int8 |] in
  let p = P.v ~ins [| P.In 0 |] ~outs:[| 0; 0; 0; 0; 0 |] in
  let call () =
    let ys = Array.init 5 (fun _ -> A.create d D.Int8 [| 1000 |]) in
    let dsts = Iarray.map (fun y -> A.Any y) (Iarray.of_array ys) in
    equal answer A.Done (Nx_metal.map p ~dsts [| Nx_kernel.Plain x |]);
    Array.iter (fun y -> equal (array string) (elements x) (elements y)) ys
  in
  let before = Nx_metal.kept_calls d in
  Fun.protect
    ~finally:(fun () -> Nx_metal.bound_kept ~launches:max_int)
    (fun () ->
      Nx_metal.bound_kept ~launches:0;
      call ();
      call ());
  equal ~msg:"past the bound" int before (Nx_metal.kept_calls d);
  call ();
  call ();
  equal ~msg:"within it" int (before + 1) (Nx_metal.kept_calls d)

(* operands.mli: keys fall in buckets by their low bits about as uniform draws
   do. The keys of copies whose destination and load share a shape, of 1 to 8
   rows of 1 to 40 elements, at byte 0, a uint8 and a float32: 320 uniform draws
   into 256 buckets fill 182.8 of them, with a standard deviation of 5.1, so
   each dtype's keys fill at least 162. *)
let key_shapes = List.init 8 (fun r -> List.init 40 (fun c -> (r + 1, c + 1)))
let uniform_least = 162

let test_hash_spread () =
  let _ = backend () in
  let d = S.device () in
  let spread (D.Any dt) =
    let size = D.bits dt / 8 in
    let x = Rig.Buffer.create d (8 * 40 * size) in
    let y = Rig.Buffer.create d (8 * 40 * size) in
    let bucket (r, c) =
      let l = L.contiguous (Iarray.of_array [| r; c |]) in
      let at b = A.v dt l (Rig.Buffer.view b ~first:0 ~length:(r * c * size)) in
      let xs = facts ~dsts:[| A.Any (at y) |] [| Plain (at x) |] in
      Nx_metal.Operands.hash (Nx_metal.Operands.key xs) land 255
    in
    let seen = Array.make 256 false in
    List.iter (List.iter (fun s -> seen.(bucket s) <- true)) key_shapes;
    let filled = Array.fold_left (fun n b -> if b then n + 1 else n) 0 seen in
    at_least ~msg:(D.name dt) int ~than:uniform_least filled
  in
  spread (D.Any D.Uint8);
  spread (D.Any D.Float32)

let test_bounds () =
  raises (Invalid_argument "Nx_metal.bound_fresh: workspace") (fun () ->
      Nx_metal.bound_fresh ~workspace:(-1) ~tickets:0);
  raises (Invalid_argument "Nx_metal.bound_fresh: tickets") (fun () ->
      Nx_metal.bound_fresh ~workspace:0 ~tickets:(kept_tickets + 1));
  raises (Invalid_argument "Nx_metal.bound_kept: launches") (fun () ->
      Nx_metal.bound_kept ~launches:(-1));
  raises (Invalid_argument "Nx_metal.bound_constants") (fun () ->
      Nx_metal.bound_constants (-1));
  raises (Invalid_argument "Nx_metal.bound_plans") (fun () ->
      Nx_metal.bound_plans (-1));
  raises (Invalid_argument "Nx_metal.bound_plans") (fun () ->
      Nx_metal.bound_plans (kept_plans + 1))

(* Kept plans *)

(* Runs [f] with no map's plan kept. *)
let unkept f =
  Fun.protect
    ~finally:(fun () -> Nx_metal.bound_plans kept_plans)
    (fun () ->
      Nx_metal.bound_plans 0;
      f ())

(* The maps whose plans a law compares: an add of the direct kernel, a copy of a
   run, a copy of a transposed operand, a padded load, a broadcast operand and
   one buffer loaded twice. *)
let planned_kinds =
  [| "add"; "copy"; "transposed copy"; "padded"; "broadcast"; "one buffer" |]

type planned_map = { kind : int; dims : int array; seed : int }

let pp_planned_map ppf c =
  Format.fprintf ppf "%s of [%s], seed %d" planned_kinds.(c.kind)
    (String.concat "; " (Array.to_list (Array.map string_of_int c.dims)))
    c.seed

let planned_map =
  let open Gen in
  with_pp pp_planned_map
    (let+ kind = int_range 0 (Array.length planned_kinds - 1)
     and+ dims = array ~size:(int_range 2 2) (int_range 0 40)
     and+ seed = nat in
     { kind; dims; seed })

(* A map of no element, which a draw reaches one case in 20, so that each run
   reaches it. *)
let planned_examples = [ { kind = 0; dims = [| 0; 5 |]; seed = 1 } ]
let f32 = D.Float32

(* A float32 array of shape [dims] on the GPU, its elements drawn from [st] in
   [-1, 1): normal numbers or zeros, which nx.metal and nx.cpu add to the same
   bits (nx_metal.mli, Subnormal numbers). *)
let values st dims =
  let n = Array.fold_left ( * ) 1 dims in
  let h =
    A.of_array f32 (Iarray.of_array dims)
      (Array.init n (fun _ -> Random.State.float st 2. -. 1.))
  in
  A.to_device (S.device ()) h

(* The program and the loads of [c]'s map into [c.dims], of arrays drawn from
   [st]. *)
let planned_call c st : P.t * Nx_kernel.load iarray =
  let x = values st c.dims and y = values st c.dims in
  let add =
    P.v ~ins:[| D.Any f32; D.Any f32 |]
      [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
      ~outs:[| 2 |]
  and copy = P.v ~ins:[| D.Any f32 |] [| P.In 0 |] ~outs:[| 0 |] in
  let flip v = Option.get (A.move (M.Permute (Iarray.of_list [ 1; 0 ])) v) in
  let row v =
    let range = { M.start = 0; count = 1; step = 1 } in
    let all = { M.start = 0; count = c.dims.(1); step = 1 } in
    let r = Option.get (A.move (M.Slice [| range; all |]) v) in
    Option.get (A.move (M.Broadcast (Iarray.of_array c.dims)) r)
  in
  match planned_kinds.(c.kind) with
  | "add" -> (add, [| Nx_kernel.Plain x; Plain y |])
  | "copy" -> (copy, [| Plain x |])
  | "transposed copy" ->
      let t = values st [| c.dims.(1); c.dims.(0) |] in
      (copy, [| Plain (flip t) |])
  | "padded" ->
      let pad =
        {
          Nx_kernel.lo = [| 1; 0 |];
          hi = [| -1; 0 |];
          interior = [| 0; 0 |];
          windows = [||];
        }
      in
      (add, [| Padded { x; fill = 3.; pad }; Plain y |])
  | "broadcast" when c.dims.(0) > 0 -> (add, [| Plain x; Plain (row y) |])
  | "broadcast" -> (add, [| Plain x; Plain y |])
  | _ -> (add, [| Plain x; Plain x |])

let launch_of (t : Plan.traced) =
  let x, y, z = t.groups in
  Printf.sprintf "%s (%d, %d, %d) resident %d%s %s"
    (Nx_metal.Kernels.name t.kernel)
    x y z t.resident
    (if t.tickets then " tickets" else "")
    (String.escaped (Bytes.to_string t.words))

(* nx.cpu's bits of [p] over [loads], each copied to the host, into fresh host
   destinations of [dsts]'s dtypes and shapes. *)
let cpu_bits p (dsts : A.any iarray) (loads : Nx_kernel.load iarray) =
  let host (l : Nx_kernel.load) : Nx_kernel.load =
    match l with
    | Plain x -> Plain (A.to_device Rig.host x)
    | Padded { x; fill; pad } ->
        Padded { x = A.to_device Rig.host x; fill; pad }
  in
  let fresh (A.Any y) =
    A.Any (A.create Rig.host (A.dtype y) (L.shape (A.layout y)))
  in
  let zs = Iarray.map fresh dsts in
  equal answer A.Done (Nx_cpu.map p ~dsts:zs (Iarray.map host loads));
  Iarray.to_array (Iarray.map Support.elements zs)

let dst_bits (dsts : A.any iarray) =
  Iarray.to_array (Iarray.map Support.elements dsts)

(* plan.mli's replay: a map traces the same launches and gives nx.cpu's bits
   where it plans, where it takes the plan an earlier call of its program and
   facts made, of other arrays and values, and with no plan kept. *)
let law_planned c =
  let d = S.device () in
  let p, loads = planned_call c (Random.State.make [| c.seed |]) in
  (* The hit's operands: other arrays and values, of the same facts. *)
  let loads' = snd (planned_call c (Random.State.make [| c.seed; 1 |])) in
  let z () : A.any iarray =
    [| A.Any (A.create d f32 (Iarray.of_array c.dims)) |]
  in
  let want = cpu_bits p (z ()) loads and want' = cpu_bits p (z ()) loads' in
  let call p loads =
    let dsts = z () in
    let a, traced = Plan.traced (fun () -> Nx_metal.map p ~dsts loads) in
    equal answer A.Done a;
    (traced, dst_bits dsts)
  in
  let off, off_bits = unkept (fun () -> call p loads) in
  (* An epoch of no plan kept, so the first call plans. *)
  Nx_metal.bound_plans kept_plans;
  let miss, miss_bits = call p loads in
  let hit, hit_bits = call p loads' in
  let launches = List.map launch_of in
  cover "no element" (Array.mem 0 c.dims);
  Array.iteri (fun k name -> cover name (c.kind = k)) planned_kinds;
  (* A call that takes a plan replays its words, which the call that made it
     traced. *)
  let found (m : Plan.traced) (h : Plan.traced) = m.words == h.words in
  cover "the second call takes the first's plan"
    (miss <> [] && List.for_all2 found miss hit);
  equal ~msg:"planned" (list string) (launches off) (launches miss);
  equal ~msg:"found" (list string) (launches off) (launches hit);
  let bits = array (array string) in
  equal ~msg:"unkept bits" bits want off_bits;
  equal ~msg:"planned bits" bits want miss_bits;
  equal ~msg:"found bits" bits want' hit_bits

(* The facts two calls of one program differ in, one a case: a load's layout
   offset, its strides, whether two loads share a buffer, a buffer's address
   modulo 16 and modulo 32, its skew (bytes before it in its memory's 32-bit
   word), whether a load overlaps another output's destination, and a padded
   load's fill. On a device whose memory has addresses, a buffer's skew is its
   residue modulo 4, so the skew case varies the residue too. The residue and
   skew cases copy, since a copy's plan reads them: a copy of runs where both
   ends lie on 16 bytes. The residue modulo 32 casts from 64 bits, whose quads
   of 32 bytes are flat only where they lie on 32 bytes (direct.metal, Casts by
   name). *)
let fact_kinds =
  [|
    "offset";
    "strides";
    "a shared buffer";
    "residue";
    "residue modulo 32";
    "skew";
    "an overlap";
    "fill";
  |]

type facts_case = {
  fact : int;
  rows : int;
  cols : int;
  flip : bool;
  fseed : int;
}

let pp_facts_case ppf c =
  Format.fprintf ppf "%s, %dx%d, %s first, seed %d" fact_kinds.(c.fact) c.rows
    c.cols
    (if c.flip then "the second variant" else "the first")
    c.fseed

let facts_case =
  let open Gen in
  with_pp pp_facts_case
    (let+ fact = int_range 0 (Array.length fact_kinds - 1)
     and+ rows = int_range 1 8
     and+ cols = int_range 1 40
     and+ flip = bool
     and+ fseed = nat in
     { fact; rows; cols; flip; fseed })

(* A case of each fact, its first variant first, so that each run reaches every
   cover of law_facts: the shared buffer's counts only where the call of one
   buffer comes first, a sixteenth of the draws. *)
let facts_examples =
  List.init (Array.length fact_kinds) (fun fact ->
      { fact; rows = 4; cols = 24; flip = false; fseed = fact })

(* The program of [c]'s two calls, one value, so that the second could take the
   first's plan. *)
let facts_prog c =
  let add =
    P.v ~ins:[| D.Any f32; D.Any f32 |]
      [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
      ~outs:[| 2 |]
  in
  match fact_kinds.(c.fact) with
  | "residue" -> P.v ~ins:[| D.Any f32 |] [| P.In 0 |] ~outs:[| 0 |]
  | "residue modulo 32" ->
      P.v ~ins:[| D.Any D.Int64 |]
        [| P.In 0; Op1 (Cast, D.Any D.Int32, 0) |]
        ~outs:[| 1 |]
  | "skew" -> P.v ~ins:[| D.Any D.Uint8 |] [| P.In 0 |] ~outs:[| 0 |]
  | "an overlap" ->
      P.v ~ins:[| D.Any f32; D.Any f32 |] [| P.In 0; In 1 |] ~outs:[| 0; 1 |]
  | _ -> add

(* Variant [v], 0 or 1, of [c]'s call: its destinations and loads, of arrays
   drawn from [st]. *)
let facts_call c v st : A.any iarray * Nx_kernel.load iarray =
  let quads = fact_kinds.(c.fact) = "residue modulo 32" in
  let dims = [| c.rows; (if quads then 4 else 1) * c.cols |] in
  let d = S.device () in
  let z dt = A.Any (A.create d dt (Iarray.of_array dims)) in
  let x () = values st dims in
  (* [dims] of [dt], contiguous, in a buffer from byte [first] of its memory,
     holding [src]'s elements where given, else drawn bits. *)
  let at_byte ?src dt ~first =
    let bytes = Array.fold_left ( * ) 1 dims * (D.bits dt / 8) in
    let whole = A.buffer (drawn st D.Uint8 [| first + bytes |]) in
    let l = L.contiguous (Iarray.of_array dims) in
    let a = A.v dt l (Rig.Buffer.view whole ~first ~length:bytes) in
    Option.iter (fun src -> A.transfer ~src ~dst:a) src;
    a
  in
  let range start count = { M.start; count; step = 1 } in
  match fact_kinds.(c.fact) with
  | "offset" ->
      let w = values st [| c.rows; c.cols + 1 |] in
      let s = M.Slice [| range 0 c.rows; range v c.cols |] in
      let a = Option.get (A.move s w) in
      ([| z f32 |], [| Nx_kernel.Plain a; Plain (x ()) |])
  | "strides" ->
      let a =
        if v = 0 then x ()
        else
          let t = values st [| c.cols; c.rows |] in
          Option.get (A.move (M.Permute (Iarray.of_list [ 1; 0 ])) t)
      in
      ([| z f32 |], [| Plain a; Plain (x ()) |])
  | "a shared buffer" ->
      let a = x () in
      ([| z f32 |], [| Plain a; Plain (if v = 0 then a else x ()) |])
  | "residue" ->
      let a = at_byte ~src:(x ()) f32 ~first:(4 * v) in
      ([| z f32 |], [| Plain a |])
  | "residue modulo 32" ->
      ([| z D.Int32 |], [| Plain (at_byte D.Int64 ~first:(16 * v)) |])
  | "skew" -> ([| z D.Uint8 |], [| Plain (at_byte D.Uint8 ~first:v) |])
  | "an overlap" ->
      (* A swap: destination 0 takes load 0 and destination 1 load 1, and in the
         second variant load 0 is a view of destination 1 and load 1 of
         destination 0, buffers of their own over the same bytes. *)
      let a = x () and b = x () in
      let alias y =
        let buf = A.buffer y in
        A.v f32 (A.layout y)
          (Rig.Buffer.view buf ~first:0 ~length:(Rig.Buffer.length buf))
      in
      let l0, l1 = if v = 0 then (x (), x ()) else (alias b, alias a) in
      ([| A.Any a; A.Any b |], [| Plain l0; Plain l1 |])
  | _ ->
      let pad =
        {
          Nx_kernel.lo = [| 1; 0 |];
          hi = [| -1; 0 |];
          interior = [| 0; 0 |];
          windows = [||];
        }
      in
      let fill = if v = 0 then 3. else 5. in
      ([| z f32 |], [| Padded { x = x (); fill; pad }; Plain (x ()) |])

(* nx_metal.mli: a map of the same program value over equal facts takes the
   earlier plan, so facts that differ take their own: two calls of one program
   whose operands differ in one fact, the first's plan kept, each give nx.cpu's
   bits. A case counts where its two calls trace different launches: there its
   fact changes the plan, so taking the first's would show. A shared buffer
   changes only the plan's refs, which no trace holds: its case counts where the
   call of one buffer comes first, whose plan, taken by the second, would read
   one load's buffer for both. The first's plan was kept for the second where a
   third call, of the first's operands, replays it: its words are the first's. A
   count of the plans kept would miss a keep that drops plans of earlier bounds,
   which a device keeps until newer ones push them out. *)
let law_facts c =
  let _ = backend () in
  let st = Random.State.make [| c.fseed |] in
  let p = facts_prog c in
  let call (dsts, loads) =
    let a, traced = Plan.traced (fun () -> Nx_metal.map p ~dsts loads) in
    equal answer A.Done a;
    traced
  in
  (* An epoch of no plan kept. *)
  Nx_metal.bound_plans kept_plans;
  let first, second = if c.flip then (1, 0) else (0, 1) in
  let calls = List.map (fun v -> facts_call c v st) [ first; second ] in
  let traces =
    List.mapi
      (fun i (dsts, loads) ->
        let want = cpu_bits p dsts loads in
        let traced = call (dsts, loads) in
        equal
          ~msg:(Printf.sprintf "call %d's bits" (i + 1))
          (array (array string))
          want (dst_bits dsts);
        traced)
      calls
  in
  let again = call (List.hd calls) in
  let found (m : Plan.traced) (h : Plan.traced) = m.words == h.words in
  let kept =
    match traces with
    | t :: _ -> t <> [] && List.equal found t again
    | [] -> false
  in
  let differ =
    match List.map (List.map launch_of) traces with
    | [ a; b ] -> a <> b
    | _ -> false
  in
  Array.iteri
    (fun k name ->
      cover name (c.fact = k);
      cover (name ^ ", the first call's plan kept") (c.fact = k && kept);
      if name = "a shared buffer" then
        cover
          (name ^ ", its call first, its plan kept")
          (c.fact = k && (not c.flip) && kept)
      else
        cover (name ^ ", the two calls plan differently") (c.fact = k && differ))
    fact_kinds

(* Calls of plans that share a submission, adds of the direct kernel at two
   lengths, so that a run one wrote holds the other's words. *)
let shared_lengths = [| 1024; 4096 |]

let shared_add =
  P.v ~ins:[| D.Any f32; D.Any f32 |]
    [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
    ~outs:[| 2 |]

(* The operands of the add at [n] elements, and nx.cpu's bits of it. *)
let shared_operands =
  lazy
    (Array.map
       (fun n ->
         let st = Random.State.make [| n |] in
         let h () =
           A.of_array f32 [| n |]
             (Array.init n (fun _ -> Random.State.float st 2. -. 1.))
         in
         let x = h () and y = h () in
         let z = A.create Rig.host f32 [| n |] in
         let loads : Nx_kernel.load iarray = [| Plain x; Plain y |] in
         equal answer A.Done (Nx_cpu.map shared_add ~dsts:[| A.Any z |] loads);
         let on v = A.to_device (S.device ()) v in
         ((on x, on y), elements z))
       shared_lengths)

(* Forcing a lazy value that another domain is forcing raises [Lazy.Undefined]:
   the domains of a stateful test force it one at a time. *)
let shared_lock = Mutex.create ()

let operands k =
  Mutex.protect shared_lock (fun () -> (Lazy.force shared_operands).(k))

let shared_call k () =
  let (x, y), _ = operands k in
  let z = A.create (S.device ()) f32 [| shared_lengths.(k) |] in
  let loads : Nx_kernel.load iarray = [| Plain x; Plain y |] in
  equal answer A.Done (Nx_metal.map shared_add ~dsts:[| A.Any z |] loads);
  elements z

(* A kept submission's run that a call of another plan wrote last holds that
   plan's words, so a call of this one writes its own: two domains adding at the
   two lengths each get nx.cpu's bits, whichever plan wrote the run each takes.
   One command draws the length of each call: each program holds a command only
   with probability 3/4, so two commands of one length each left one out of all
   4 programs about one run in 400. *)
let domains_share_runs =
  let length = Gen.int_range 0 (Array.length shared_lengths - 1) in
  let want k =
    cover "an add at the shorter length" (k = 0);
    cover "an add at the longer length" (k = 1);
    snd (operands k)
  in
  stateful ~domains:2 ~count:4
    "calls of plans sharing a submission get their bits"
    [
      command "add"
        (length @-> returns (array string))
        want
        (fun k -> shared_call k ());
    ]

(* The two lengths' adds share a submission: their launches are of one kernel,
   whose refs name the same slots. *)
let test_shared_kernel () =
  let _ = backend () in
  let names k =
    let _, traced = Plan.traced (shared_call k) in
    List.map (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel) traced
  in
  equal (list string) [ "loop_map_direct32_flat" ] (names 0);
  equal (list string) (names 0) (names 1)

(* Direct maps *)

let direct_kernel = "loop_map_direct"

(* The launches of the direct kernels, of flat indices and of tiles, of any
   operands and of 32-bit ones. *)
let directs traced =
  List.filter
    (fun (t : Plan.traced) ->
      match t.kernel with
      | Loop_map_direct | Loop_map_direct_tiled | Loop_map_direct32
      | Loop_map_direct32_tiled | Loop_map_direct32_long
      | Loop_map_direct32_long_tiled | Loop_map_direct32_flat
      | Loop_map_direct32_long_flat | Loop_map_direct16_bf16
      | Loop_map_direct16_bf16_tiled | Loop_map_direct16_bf16_long
      | Loop_map_direct16_bf16_long_tiled | Loop_map_direct16_f16
      | Loop_map_direct16_f16_tiled | Loop_map_direct16_f16_long
      | Loop_map_direct16_f16_long_tiled | Loop_map_direct8
      | Loop_map_direct8_tiled ->
          true
      | _ -> false)
    traced

(* A launch's mode word: its v and flags (kernels.h, NX_LOOP_MODE). *)
let mode_of (t : Plan.traced) =
  Int64.to_int (Bytes.get_int64_le t.words (8 * Nx_metal.Kernels.loop_mode))

(* nx_metal.mli: a direct map runs a 32-bit kernel exactly where its form takes
   each element as it is and gives its result as it is, every load and its
   destination 32 bits wide, and its threads take quads: of tiles where it takes
   tiles, of its long kinds where [long], flat where [flat] (by default, where
   it ran flat). [words32] says whether the map's form is as is at 32 bits,
   which its widths decide: the tape takes a truth only of a bool, decodes only
   a bfloat16 or a float16, and wraps or encodes only into a narrower dtype, so
   a form whose loads and destination are each of 32 bits converts none.
   Likewise it runs a narrow float's kernel exactly where every load and its
   destination are that float's, [words16], which the tape decodes and encodes,
   and its threads take quads; and the kernel of bytes exactly where every load
   and its destination are 8 bits wide, [words8], its threads take quads and its
   kind is not long. *)
let check_direct32 ?flat ?words16 ?(words8 = false) ~words32 ~long traced =
  List.iter
    (fun (t : Plan.traced) ->
      let m = mode_of t in
      let quads = m land Nx_metal.Kernels.loop_quads <> 0 in
      let tiled = m land Nx_metal.Kernels.loop_tiled <> 0 in
      let lanes =
        match words16 with
        | Some (D.Any d) when quads ->
            Some
              ("loop_map_direct16_"
              ^ if D.Any d = D.Any D.Bfloat16 then "bf16" else "f16")
        | Some _ | None ->
            if words32 && quads then Some "loop_map_direct32"
            else if words8 && quads && not long then Some "loop_map_direct8"
            else None
      in
      let b32 = "loop_map_direct32" in
      let kind k = if long && k <> "loop_map_direct8" then k ^ "_long" else k in
      let name = Nx_metal.Kernels.name t.kernel in
      let flat =
        match flat with Some f -> f | None -> name = kind b32 ^ "_flat"
      in
      let want =
        match (lanes, tiled) with
        | Some k, false -> if flat && k = b32 then kind k ^ "_flat" else kind k
        | Some k, true -> kind k ^ "_tiled"
        | None, false -> direct_kernel
        | None, true -> direct_kernel ^ "_tiled"
      in
      equal ~msg:"the direct kernel" string want
        (Nx_metal.Kernels.name t.kernel))
    (directs traced)

(* nx_metal.mli: whether [node], computing at [dt], is a long kind of a 32-bit
   direct map. *)
let long_kind (D.Any dt) (node : P.node) =
  let real = D.Any dt = D.Any D.Float32 in
  match node with
  | Op1 (Unary k, _, _) -> (
      match k with
      | Recip -> true
      | Sqrt | Exp | Exp2 | Log | Log2 | Log1p | Expm1 | Sin | Cos | Tan | Asin
      | Acos | Atan | Sinh | Cosh | Tanh | Erf ->
          real
      | _ -> false)
  | Op2 (Binary k, _, _) -> (
      match k with
      | Mod | Pow -> true
      | Fdiv | Atan2 -> real
      | Idiv -> not real
      | _ -> false)
  | _ -> false

(* Whether a launch of [traced] is [k]'s. *)
let ran k traced = List.exists (fun (t : Plan.traced) -> t.kernel = k) traced

(* The dtypes of whole bytes of at most 32 bits, which a direct map takes. *)
let direct_dtypes =
  D.
    [
      Any Float32;
      Any Bfloat16;
      Any Float16;
      Any Int32;
      Any Uint32;
      Any Int16;
      Any Uint16;
      Any Int8;
      Any Uint8;
      Any Bool;
    ]

(* The dtypes a Metal reduction takes that a direct map does not. *)
let one_term_dtypes =
  D.
    [
      Any Int64;
      Any Uint64;
      Any Complex64;
      Any Int4;
      Any Uint4;
      Any Bit;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
    ]

let le16 x = String.init 2 (fun i -> Char.chr ((x lsr (8 * i)) land 0xff))

(* [dt]'s specials: ±0, infinities, NaNs quiet and signalling with payloads,
   subnormals, the largest finite values and integer extremes. *)
let specials (D.Any d) =
  match d with
  | D.Bfloat16 ->
      List.map le16
        [
          0x0000;
          0x8000;
          0x7f80;
          0xff80;
          0x7fc0;
          0xffc1;
          0x7f81;
          0x0001;
          0x807f;
          0x7f7f;
          0x3f80;
          0x0080;
        ]
  | D.Float16 ->
      List.map le16
        [
          0x0000;
          0x8000;
          0x7c00;
          0xfc00;
          0x7e00;
          0xfe01;
          0x7c01;
          0x0001;
          0x83ff;
          0x7bff;
          0x3c00;
          0x0400;
        ]
  | d -> Kernels_cases.specials d

(* A fresh host array of [dt] and shape [s], each element a special half the
   time, else drawn bytes. *)
let host_specials st (D.Any d as dt) s =
  let x = A.create Rig.host d (Iarray.of_array s) in
  let w = D.bytes d 1 and n = Array.fold_left ( * ) 1 s in
  let sp = Array.of_list (specials dt) in
  let one () =
    if Array.length sp > 0 && Random.State.bool st then
      sp.(Random.State.int st (Array.length sp))
    else random_bytes st w
  in
  (* A dtype narrower than a byte packs its elements: random bytes draw every
     code. *)
  let bytes =
    if D.bits d < 8 then random_bytes st (((n * D.bits d) + 7) / 8)
    else String.concat "" (List.init n (fun _ -> one ()))
  in
  Rig.Buffer.blit_from_string bytes 0 (A.buffer x) 0 (String.length bytes);
  A.Any x

(* As {!host_specials}, on the GPU. *)
let special_drawn st dt s =
  let (A.Any h) = host_specials st dt s in
  A.Any (A.to_device (S.device ()) h)

(* How a direct map's operand lies over its loop: contiguous; its rows [pad]
   elements apart past their last; every other element of its last axis; its
   first axis reversed; its last axis reversed; broadcast along an axis; or
   contiguous from position [k] of its buffer. *)
type view =
  | Whole
  | Rows of int
  | Stepped
  | Flipped
  | Reversed
  | Spread of int
  | Shifted of int

let pp_view ppf = function
  | Whole -> Format.pp_print_string ppf "whole"
  | Rows p -> Format.fprintf ppf "rows +%d" p
  | Stepped -> Format.pp_print_string ppf "stepped"
  | Flipped -> Format.pp_print_string ppf "flipped"
  | Reversed -> Format.pp_print_string ppf "reversed"
  | Spread a -> Format.fprintf ppf "spread %d" a
  | Shifted k -> Format.fprintf ppf "shifted %d" k

(* The shape of the array that [view] of shape [s] lies over. *)
let base_shape s view =
  let r = Array.length s in
  let grown f =
    let s' = Array.copy s in
    s'.(r - 1) <- f s.(r - 1);
    s'
  in
  match view with
  | Rows pad when r > 0 -> grown (fun e -> e + pad)
  | Stepped when r > 0 -> grown (fun e -> 2 * e)
  | Spread a when r > 0 ->
      let s' = Array.copy s in
      s'.(a mod r) <- 1;
      s'
  | Shifted k -> [| k + Array.fold_left ( * ) 1 s |]
  | Whole | Rows _ | Stepped | Flipped | Reversed | Spread _ -> s

(* [view] of shape [s] over [b], an array of {!base_shape}[ s view]. *)
let on_view (A.Any x as b) s view =
  let r = Array.length s in
  let all e = { M.start = 0; count = e; step = 1 } in
  let slice rs = A.Any (Option.get (A.move (M.Slice (Iarray.of_array rs)) x)) in
  match view with
  | Rows _ when r > 0 -> slice (Array.map all s)
  | Stepped when r > 0 ->
      let rs = Array.map all s in
      rs.(r - 1) <- { M.start = 1; count = s.(r - 1); step = 2 };
      slice rs
  | Flipped when r > 0 && s.(0) > 0 ->
      let rs = Array.map all s in
      rs.(0) <- { M.start = s.(0) - 1; count = s.(0); step = -1 };
      slice rs
  | Reversed when r > 0 && s.(r - 1) > 0 ->
      let rs = Array.map all s in
      rs.(r - 1) <- { M.start = s.(r - 1) - 1; count = s.(r - 1); step = -1 };
      slice rs
  | Spread _ when r > 0 ->
      A.Any (Option.get (A.move (M.Broadcast (Iarray.of_array s)) x))
  | Shifted k ->
      let n = Array.fold_left ( * ) 1 s in
      let (A.Any run) = slice [| { M.start = k; count = n; step = 1 } |] in
      A.Any (Option.get (A.move (M.Reshape (Iarray.of_array s)) run))
  | Whole | Rows _ | Stepped | Flipped | Reversed | Spread _ -> b

(* [x] of shape [s], over [view], of the array [base] makes; and that array. *)
let viewed base dt s view =
  let b = base dt (base_shape s view) in
  (on_view b s view, b)

(* A program of one node over its loads: its name, its loads' dtypes and its
   node, which is node [Array.length ins]. *)
type node = { name : string; ins : D.any array; node : P.node }

(* The classes of one-node programs over [dt]'s loads that P takes: the unary
   kinds; the casts into another of [direct_dtypes]; the binary kinds of two
   loads; of one load with itself; the comparisons; Fma; and Where. *)
let direct_classes (D.Any d as dt) =
  let unary =
    List.filter_map
      (fun (u, n) ->
        if P.accepts1 (Unary u) d d then
          Some { name = n; ins = [| dt |]; node = P.Op1 (Unary u, dt, 0) }
        else None)
      Kernels_cases.unaries
  in
  let casts =
    List.filter_map
      (fun (D.Any y as into) ->
        if into <> dt && P.accepts1 Cast d y then
          Some
            {
              name = "cast to " ^ D.name y;
              ins = [| dt |];
              node = P.Op1 (Cast, into, 0);
            }
        else None)
      direct_dtypes
  in
  let twos ~self ~compare =
    List.filter_map
      (fun k ->
        let is_compare = match k with P.Compare _ -> true | _ -> false in
        if is_compare = compare && P.accepts2 k d then
          let n = Kernels_cases.name2 k in
          Some
            (if self then
               {
                 name = n ^ " of itself";
                 ins = [| dt |];
                 node = P.Op2 (k, 0, 0);
               }
             else { name = n; ins = [| dt; dt |]; node = P.Op2 (k, 0, 1) })
        else None)
      Kernels_cases.op2s
  in
  let fma =
    if P.accepts3 Fma d d then
      [ { name = "fma"; ins = [| dt; dt; dt |]; node = P.Op3 (Fma, 0, 1, 2) } ]
    else []
  in
  let where =
    if P.accepts3 Where D.Bool d then
      [
        {
          name = "where";
          ins = [| D.Any D.Bool; dt; dt |];
          node = P.Op3 (Where, 0, 1, 2);
        };
      ]
    else []
  in
  List.filter
    (fun c -> c <> [])
    [
      unary;
      casts;
      twos ~self:false ~compare:false;
      twos ~self:true ~compare:false;
      twos ~self:false ~compare:true;
      fma;
      where;
    ]

(* The casts between float32 and a narrow float, which a direct map decodes or
   encodes. *)
let narrow_casts =
  List.map
    (fun (dt, (D.Any y as into)) ->
      {
        name = "cast to " ^ D.name y;
        ins = [| dt |];
        node = P.Op1 (Cast, into, 0);
      })
    D.
      [
        (Any Float32, Any Bfloat16);
        (Any Float32, Any Float16);
        (Any Bfloat16, Any Float32);
        (Any Float16, Any Float32);
      ]

let out_dtype n =
  match n.node with
  | P.Op1 (_, y, _) -> y
  | Op2 (Compare _, _, _) -> D.Any D.Bool
  | Op2 (Binary _, _, _) | Op3 (Fma, _, _, _) -> n.ins.(0)
  | Op3 (Where, _, _, _) -> n.ins.(1)
  | In _ | Coord _ | Const _ -> invalid_arg "out_dtype"

type direct = {
  dt : D.any;
  form : node;
  shape : int array;
  views : view array;
  dview : view;
  bases : int array;
  dseed : int;
}

let pp_direct ppf c =
  let (D.Any d) = c.dt in
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf
    "%s at %s over [%s], loads %a into %a, from bytes [%s], seed %d" c.form.name
    (D.name d) (ints c.shape)
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
       pp_view)
    (Array.to_list c.views) pp_view c.dview (ints c.bases) c.dseed

(* Shapes of no element and of one, and of up to 3 axes whose last extent is a
   multiple of 4 or not, of several groups' elements at most. *)
let direct_shape =
  let open Gen in
  let extent =
    of_list ~pp:Format.pp_print_int
      [ 1; 2; 3; 4; 5; 7; 8; 9; 12; 16; 33; 64; 257; 1000 ]
  in
  let drawn =
    let* r = int_range 1 3 in
    such_that
      (fun s -> Array.fold_left ( * ) 1 s <= 1 lsl 14)
      (array ~size:(constant r) extent)
  in
  frequency
    [
      (1, of_list [ [||]; [| 0 |]; [| 1 |]; [| 1; 1 |]; [| 3; 0 |]; [| 1; 4 |] ]);
      (6, drawn);
    ]

let direct =
  let open Gen in
  let view =
    frequency
      [
        (4, constant Whole);
        (2, map (fun p -> Rows p) (int_range 1 3));
        (1, constant Stepped);
        (1, constant Flipped);
        (2, map (fun a -> Spread a) (int_range 0 2));
        (2, map (fun k -> Shifted k) (int_range 1 5));
      ]
  in
  let dview =
    frequency
      [
        (4, constant Whole);
        (1, map (fun p -> Rows p) (int_range 1 3));
        (1, constant Stepped);
        (2, map (fun k -> Shifted k) (int_range 1 5));
      ]
  in
  (* The byte of a fresh buffer an operand's buffer starts at: half the time 0,
     else one of no multiple of 16, which {!direct_run} takes down to a multiple
     of the operand's element. *)
  let base =
    frequency [ (1, constant 0); (1, of_list [ 1; 2; 3; 4; 6; 8; 12 ]) ]
  in
  (* The long kinds of int8 and uint8, which the direct kernel runs in place of
     the kernel of bytes: drawn alone a case in 13, as the class draw below
     reaches them in under 1% of cases. *)
  let long_bytes =
    List.concat_map
      (fun (D.Any d as dt) ->
        List.filter_map
          (fun k ->
            if P.accepts2 (Binary k) d then
              Some
                {
                  name = Kernels_cases.name2 (Binary k);
                  ins = [| dt; dt |];
                  node = P.Op2 (Binary k, 0, 1);
                }
            else None)
          P.[ Idiv; Mod; Pow ])
      [ D.Any D.Int8; D.Any D.Uint8 ]
  in
  (* A dtype, a class of its programs and one of them, or half the time in place
     of a cast, a cast between float32 and a narrow float. *)
  let classed =
    (* 32-bit dtypes twice as often, so that flat maps of 32 bits reach each of
       their covers. *)
    let* dt =
      frequency
        (List.map
           (fun (D.Any d as dt) ->
             ((if D.bits d = 32 then 2 else 1), constant dt))
           direct_dtypes)
    in
    let classes = direct_classes dt in
    let* k = int_range 0 (List.length classes - 1) in
    let forms = of_list (List.nth classes k) in
    match List.hd (List.nth classes k) with
    | { node = P.Op1 (Cast, _, _); _ } ->
        frequency [ (1, forms); (1, of_list narrow_casts) ]
    | _ -> forms
  in
  let form = frequency [ (12, classed); (1, of_list long_bytes) ] in
  with_pp pp_direct
    (let* form = form in
     let dt = form.ins.(Array.length form.ins - 1) in
     let+ shape = direct_shape
     and+ views = array ~size:(constant (Array.length form.ins)) view
     and+ dview = dview
     and+ bases = array ~size:(constant (Array.length form.ins + 1)) base
     and+ dseed = nat in
     { dt; form; shape; views; dview; bases; dseed })

(* The byte, at most [b], at which [dt]'s elements may start. *)
let base_of (D.Any d) b = b / D.bytes d 1 * D.bytes d 1

(* [x] and [whole], which share a buffer, over a copy of its bytes from byte
   [at] of a fresh buffer on the GPU. *)
let rebased at (A.Any x) (A.Any whole) =
  let src = A.buffer whole in
  let n = Rig.Buffer.length src in
  let dst =
    Rig.Buffer.view
      (Rig.Buffer.create (S.device ()) (at + n))
      ~first:at ~length:n
  in
  Rig.Buffer.copy ~src ~dst;
  ( A.Any (A.v (A.dtype x) (A.layout x) dst),
    A.Any (A.v (A.dtype whole) (A.layout whole) dst) )

(* [c]'s map, the direct kernel taken where [direct], from operands and a
   destination's array drawn from [c]'s seed: its answer, the bytes of the
   destination's array and the launches planned. *)
let direct_run c ~direct =
  let st = Random.State.make [| c.dseed |] in
  let base dt s = special_drawn st dt s in
  let m = Array.length c.form.ins in
  let placed dt view at =
    let x, whole = viewed base dt c.shape view in
    rebased (base_of dt at) x whole
  in
  let loads =
    Array.mapi
      (fun k dt ->
        let A.Any x, _ = placed dt c.views.(k) c.bases.(k) in
        Nx_kernel.Plain x)
      c.form.ins
  in
  let dst, whole = placed (out_dtype c.form) c.dview c.bases.(m) in
  let strides (A.Any x) = Iarray.to_array (L.strides (A.layout x)) in
  let loaded = function
    | Nx_kernel.Plain x -> strides (A.Any x)
    | Padded _ -> invalid_arg "direct_run: a padded load"
  in
  let all = strides dst :: List.map loaded (Array.to_list loads) in
  let p =
    P.v
      ~ins:(Iarray.of_array c.form.ins)
      (Iarray.of_array
         (Array.append (Array.init m (fun k -> P.In k)) [| c.form.node |]))
      ~outs:[| m |]
  in
  let answer, traced =
    Fun.protect
      ~finally:(fun () -> Nx_metal.bound_direct true)
      (fun () ->
        Nx_metal.bound_direct direct;
        Plan.traced (fun () ->
            Nx_metal.map p ~dsts:[| dst |] (Iarray.of_array loads)))
  in
  (answer, Support.elements whole, traced, all)

let numel s = Array.fold_left ( * ) 1 s

(* nx_metal.mli: whether a map of [shape] over operands of [strides], the
   destination's first, is flat: element [i] of the destination, in the order of
   its indices, lies [i] elements past its first, and element [i] of each load
   [i] past its first or at it. *)
let flat_walk shape strides =
  let n = numel shape in
  let at s i =
    let p = ref 0 and i = ref i in
    for a = Array.length shape - 1 downto 0 do
      p := !p + (!i mod shape.(a) * s.(a));
      i := !i / shape.(a)
    done;
    !p
  in
  let every f = Seq.for_all f (Seq.init n Fun.id) in
  let dst = List.hd strides and loads = List.tl strides in
  every (fun i -> at dst i = i)
  && List.for_all
       (fun s -> every (fun i -> at s i = i) || every (fun i -> at s i = 0))
       loads

(* The place codes of the slots of [traced]'s direct launches (kernels.h,
   NX_DIRECT_PLACES). *)
let place_codes traced =
  let module K = Nx_metal.Kernels in
  List.concat_map
    (fun (t : Plan.traced) ->
      let f = Int64.to_int (Bytes.get_int64_le t.words (8 * K.loop_direct)) in
      List.init 4 (fun k -> (f lsr (K.direct_places + (2 * k))) land 3))
    (directs traced)

(* Whether an operand of [dt] over [view] from byte [b] of its buffer is
   contiguous, its first element's offset a multiple of a quad's bytes and its
   buffer's first byte not: its quads lie on their size by its offset alone, and
   not by its address. *)
let off_by_base (D.Any d as dt) view b =
  let w = D.bytes d 1 in
  let on k = k * w mod (4 * w) = 0 && base_of dt b mod (4 * w) <> 0 in
  match view with
  | Whole -> on 0
  | Shifted k -> on k
  | Rows _ | Stepped | Flipped | Reversed | Spread _ -> false

(* Maps that run the 32-bit kernels with descriptors: their loads' rows apart,
   so no walk of theirs is one axis; an add and an exp, of the cheap kinds and
   the long ones. Then maps that the draws reach a few times in 300 each: a flat
   add of one axis whose second load is one element broadcast, a slot that reads
   its first element at every index; at bfloat16 and float16, an add and an exp
   of 4099 elements, which run their codec's kernels; and of bytes, a uint8
   product, an int8 comparison and where, which run the kernel of bytes, and an
   int8 Idiv, a long kind, which runs the direct kernel. *)
let direct_examples =
  let f32 = D.Any D.Float32 and bf16 = D.Any D.Bfloat16 in
  let f16 = D.Any D.Float16 and i8 = D.Any D.Int8 and u8 = D.Any D.Uint8 in
  let c ?(dt = f32) form shape views =
    {
      dt;
      form;
      shape;
      views;
      dview = Whole;
      bases = Array.make (Array.length views + 1) 0;
      dseed = 1;
    }
  in
  let two name dt k = { name; ins = [| dt; dt |]; node = P.Op2 (k, 0, 1) } in
  let exp dt =
    { name = "exp"; ins = [| dt |]; node = P.Op1 (Unary Exp, dt, 0) }
  in
  let quads dt form =
    c ~dt form [| 4099 |] (Array.map (fun _ -> Whole) form.ins)
  in
  [
    c (two "add" f32 (Binary Add)) [| 3; 8 |] [| Rows 1; Whole |];
    c (exp f32) [| 3; 8 |] [| Rows 1 |];
    c (two "add" f32 (Binary Add)) [| 1000 |] [| Whole; Spread 0 |];
    (* Several groups, the last partial: 5000 elements over rows apart, and 4099
       in a flat walk. *)
    c (two "add" f32 (Binary Add)) [| 5; 1000 |] [| Rows 1; Whole |];
    c (two "add" f32 (Binary Add)) [| 4099 |] [| Whole; Whole |];
    quads bf16 (two "add" bf16 (Binary Add));
    quads bf16 (exp bf16);
    quads f16 (two "add" f16 (Binary Add));
    quads f16 (exp f16);
    quads u8 (two "mul" u8 (Binary Mul));
    quads i8 (two "less" i8 (Compare Less));
    quads i8
      {
        name = "where";
        ins = [| D.Any D.Bool; i8; i8 |];
        node = P.Op3 (Where, 0, 1, 2);
      };
    quads i8 (two "idiv" i8 (Binary Idiv));
  ]

(* Maps whose covers draws reach in 1 to 7 % of the cases, past those
   direct_examples names: the kernel of bytes at int8 and at booleans, over 4099
   elements; a flat 32-bit Exp of two axes merged; flat adds over a load and
   into a destination past their buffers' first elements, and over an operand
   whose offset lies on its quads and whose buffer does not; maps of no element
   and of one; and narrow floats' maps of a stepped load and of rows apart,
   without quads. *)
let direct_rare =
  let f32 = D.Any D.Float32 and i8 = D.Any D.Int8 in
  let b = D.Any D.Bool and bf16 = D.Any D.Bfloat16 and f16 = D.Any D.Float16 in
  let two name dt k = { name; ins = [| dt; dt |]; node = P.Op2 (k, 0, 1) } in
  let add dt = two "add" dt (Binary Add) in
  let c ?views ?(dview = Whole) ?bases form shape =
    let m = Array.length form.ins in
    {
      dt = form.ins.(m - 1);
      form;
      shape;
      views = Option.value views ~default:(Array.make m Whole);
      dview;
      bases = Option.value bases ~default:(Array.make (m + 1) 0);
      dseed = 3;
    }
  in
  [
    c (two "maximum" i8 (Binary Maximum)) [| 4099 |];
    c (two "and" b (Binary And)) [| 4099 |];
    c
      { name = "exp"; ins = [| f32 |]; node = P.Op1 (Unary Exp, f32, 0) }
      [| 3; 1000 |];
    c ~views:[| Shifted 1; Whole |] ~dview:(Shifted 2) (add f32) [| 1000 |];
    c ~bases:[| 4; 0; 0 |] (add f32) [| 1000 |];
    c (add f32) [| 3; 0 |];
    c (add f32) [| 1; 1 |];
    c ~views:[| Stepped; Whole |] (add bf16) [| 3; 8 |];
    c ~views:[| Rows 1; Whole |] (add f16) [| 3; 8 |];
  ]

(* nx_metal.mli: a narrow float's direct map of quads runs its codec's kernel.
   Its operands and destination start 2 or 6 bytes into their buffers, off a
   quad's 8, over a length off a multiple of 4, so its quads are read and
   written element by element: each has the tape's bits. *)
let test_direct16_shifted () =
  let _ = backend () in
  List.iter
    (fun (dt, kernel) ->
      List.iter
        (fun (a, b, o) ->
          let form =
            {
              name = "add";
              ins = [| dt; dt |];
              node = P.Op2 (Binary Add, 0, 1);
            }
          in
          let c =
            {
              dt;
              form;
              shape = [| 4099 |];
              views = [| Shifted a; Shifted b |];
              dview = Shifted o;
              bases = [| 0; 0; 0 |];
              dseed = (100 * a) + (10 * b) + o;
            }
          in
          let msg = Format.asprintf "%a" pp_direct c in
          let a, want, _, _ = direct_run c ~direct:false in
          let a', got, traced, _ = direct_run c ~direct:true in
          equal ~msg answer a a';
          equal ~msg (array string) want got;
          equal ~msg (list string) [ kernel ]
            (List.map
               (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
               (directs traced)))
        [ (1, 3, 0); (3, 1, 1); (0, 0, 3) ])
    [
      (D.Any D.Bfloat16, "loop_map_direct16_bf16");
      (D.Any D.Float16, "loop_map_direct16_f16");
    ]

(* nx_metal.mli: a direct map of bytes in quads runs the kernel of bytes but for
   its long kinds, which run the direct kernel. Its forms take int8s whose sign
   decides the result (Maximum, Less), products past 8 bits at int8 and uint8,
   booleans (And, Where's condition), an int8 Idiv, a long kind, over 4099
   elements of drawn bytes, the operands and destination 0 to 3 bytes into their
   buffers, so that some quads lie on 4 bytes and some do not: each has the
   tape's bits. *)
let test_direct8_kinds () =
  let _ = backend () in
  let i8 = D.Any D.Int8 and u8 = D.Any D.Uint8 and b = D.Any D.Bool in
  let two name dt k = { name; ins = [| dt; dt |]; node = P.Op2 (k, 0, 1) } in
  let where dt =
    { name = "where"; ins = [| b; dt; dt |]; node = P.Op3 (Where, 0, 1, 2) }
  in
  List.iter
    (fun (dt, form, kernel) ->
      List.iter
        (fun (a, o) ->
          let c =
            {
              dt;
              form;
              shape = [| 4099 |];
              views = Array.map (fun _ -> Shifted a) form.ins;
              dview = Shifted o;
              bases = Array.make (Array.length form.ins + 1) 0;
              dseed = (10 * a) + o;
            }
          in
          let msg = Format.asprintf "%a" pp_direct c in
          let a, want, _, _ = direct_run c ~direct:false in
          let a', got, traced, _ = direct_run c ~direct:true in
          equal ~msg answer a a';
          equal ~msg (array string) want got;
          equal ~msg (list string) [ kernel ]
            (List.map
               (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
               (directs traced)))
        [ (0, 0); (1, 3); (3, 2); (2, 1) ])
    [
      (i8, two "maximum" i8 (Binary Maximum), "loop_map_direct8");
      (i8, two "less" i8 (Compare Less), "loop_map_direct8");
      (i8, two "mul" i8 (Binary Mul), "loop_map_direct8");
      (u8, two "mul" u8 (Binary Mul), "loop_map_direct8");
      (b, two "and" b (Binary And), "loop_map_direct8");
      (i8, where i8, "loop_map_direct8");
      (i8, two "idiv" i8 (Binary Idiv), direct_kernel);
    ]

(* A map of a direct form gives the tape's answer and bits, NaN payloads
   included, and [bound_direct false] makes it run the tape: over operands at
   any first position, in buffers that start at any byte their elements may. *)
let law_direct c =
  let (D.Any d) = c.dt in
  let a, want, tape, _ = direct_run c ~direct:false in
  let a', got, direct, strides = direct_run c ~direct:true in
  equal ~msg:"the tape alone" (list string) []
    (List.map
       (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
       (directs tape));
  equal answer a a';
  equal (array string) want got;
  let words32 =
    Array.for_all (fun (D.Any d) -> D.bits d = 32) c.form.ins
    &&
    let (D.Any o) = out_dtype c.form in
    D.bits o = 32
  in
  (* The loads the form reads: a load with itself reads one of two. *)
  let read =
    match c.form.node with
    | P.Op1 (_, _, a) -> [ a ]
    | Op2 (_, a, b) -> [ a; b ]
    | Op3 (_, a, b, e) -> [ a; b; e ]
    | _ -> List.init (Array.length c.form.ins) Fun.id
  in
  let flat =
    flat_walk c.shape
      (List.hd strides
      :: List.filteri (fun k _ -> List.mem k read) (List.tl strides))
  in
  let narrow16 dt = dt = D.Any D.Bfloat16 || dt = D.Any D.Float16 in
  let words16 =
    let o = out_dtype c.form in
    if narrow16 o && Array.for_all (( = ) o) c.form.ins then Some o else None
  in
  (* A narrow float's kinds compute at float32. *)
  let long =
    long_kind
      (if words16 = None then c.form.ins.(0) else D.Any D.Float32)
      c.form.node
  in
  let words8 =
    Array.for_all (fun (D.Any d) -> D.bits d = 8) c.form.ins
    &&
    let (D.Any o) = out_dtype c.form in
    D.bits o = 8
  in
  check_direct32 ~flat ?words16 ~words8 ~words32 ~long direct;
  let took = directs direct <> [] in
  let narrow = D.Any d = D.Any D.Bfloat16 || D.Any d = D.Any D.Float16 in
  let node = c.form.node in
  cover "the direct kernel" took;
  cover "the 32-bit kernel" (ran Loop_map_direct32 direct);
  cover "the 32-bit kernel of long kinds" (ran Loop_map_direct32_long direct);
  cover "the flat 32-bit kernel" (ran Loop_map_direct32_flat direct);
  cover "the flat 32-bit kernel of long kinds"
    (ran Loop_map_direct32_long_flat direct);
  cover "a flat walk of 32 bits over several axes merged"
    (words32 && flat
    && Array.length c.shape > 1
    && Array.for_all (fun e -> e > 1) c.shape);
  cover "the bfloat16 kernel" (ran Loop_map_direct16_bf16 direct);
  cover "the bfloat16 kernel of long kinds"
    (ran Loop_map_direct16_bf16_long direct);
  cover "the float16 kernel" (ran Loop_map_direct16_f16 direct);
  cover "the float16 kernel of long kinds"
    (ran Loop_map_direct16_f16_long direct);
  cover "a narrow float's direct map without quads"
    (words16 <> None && ran Loop_map_direct direct);
  let bytes = ran Loop_map_direct8 direct in
  cover "the kernel of bytes" bytes;
  cover "the kernel of bytes at int8" (bytes && D.Any d = D.Any D.Int8);
  cover "the kernel of bytes at uint8" (bytes && D.Any d = D.Any D.Uint8);
  cover "the kernel of bytes at booleans" (bytes && D.Any d = D.Any D.Bool);
  cover "the kernel of bytes of a comparison"
    (bytes && match node with P.Op2 (Compare _, _, _) -> true | _ -> false);
  cover "the kernel of bytes of a where"
    (bytes && match node with P.Op3 (Where, _, _, _) -> true | _ -> false);
  cover "bytes: int8 under Maximum, Minimum or a comparison"
    (bytes && D.Any d = D.Any D.Int8
    &&
    match node with
    | P.Op2 ((Binary (Maximum | Minimum) | Compare _), _, _) -> true
    | _ -> false);
  cover "bytes: an Add or Mul of int8 or uint8, its results wrapped"
    (bytes
    && (D.Any d = D.Any D.Int8 || D.Any d = D.Any D.Uint8)
    && match node with P.Op2 (Binary (Add | Mul), _, _) -> true | _ -> false);
  cover "a long kind of bytes in the direct kernel"
    (words8 && long && ran Loop_map_direct direct);
  cover "a direct map of bytes without quads"
    (words8 && (not long) && ran Loop_map_direct direct);
  cover "a 32-bit direct map without quads"
    (words32 && ran Loop_map_direct direct);
  cover "the direct kernel at a narrow float" (took && narrow);
  cover "the direct kernel at a narrow integer"
    (took && D.bits d < 32 && (D.is D.Signed d || D.is D.Unsigned d));
  cover "the direct kernel at booleans" (took && D.Any d = D.Any D.Bool);
  cover "the direct kernel of a comparison"
    (took && match node with P.Op2 (Compare _, _, _) -> true | _ -> false);
  cover "the direct kernel of a where"
    (took && match node with P.Op3 (Where, _, _, _) -> true | _ -> false);
  cover "the direct kernel of an fma"
    (took && match node with P.Op3 (Fma, _, _, _) -> true | _ -> false);
  cover "the direct kernel of a load with itself"
    (took && match node with P.Op2 (_, 0, 0) -> true | _ -> false);
  cover "the direct kernel over rows apart by no multiple of 4"
    (took && Array.exists (function Rows _ -> true | _ -> false) c.views);
  cover "the direct kernel over a broadcast load"
    (took && Array.exists (function Spread _ -> true | _ -> false) c.views);
  cover "the direct kernel over a reversed load"
    (took && Array.mem Flipped c.views);
  cover "the direct kernel into a strided destination" (took && c.dview <> Whole);
  cover "the direct kernel over several groups" (took && numel c.shape > 1024);
  cover "the direct kernel over several groups, the last partial"
    (took && numel c.shape > 1024 && numel c.shape mod 1024 <> 0);
  cover "the direct kernel of one element" (took && numel c.shape = 1);
  let module K = Nx_metal.Kernels in
  let codes = place_codes direct in
  let flat = took && words32 && flat in
  cover "a flat slot whose quads lie on their size"
    (List.mem K.direct_flat_on codes);
  cover "a flat slot whose quads do not" (List.mem K.direct_flat codes);
  cover "a slot whose every index reads its first element"
    (List.mem K.direct_first codes);
  cover "a flat map over a load past its buffer's first element"
    (flat && Array.exists (function Shifted _ -> true | _ -> false) c.views);
  cover "a flat map into a destination past its buffer's first element"
    (flat && match c.dview with Shifted _ -> true | _ -> false);
  cover "a flat map over several groups, the last partial"
    (flat && numel c.shape > 1024 && numel c.shape mod 1024 <> 0);
  let m = Array.length c.form.ins in
  cover
    "a flat map over an operand whose offset lies on its quads and whose \
     buffer does not"
    (flat
    && (off_by_base (out_dtype c.form) c.dview c.bases.(m)
       || Array.exists Fun.id
            (Array.mapi
               (fun k dt -> off_by_base dt c.views.(k) c.bases.(k))
               c.form.ins)));
  cover "no element" (numel c.shape = 0)

(* nx_metal.mli: a direct map's loop has at most 8 axes. An add of [r] axes of 2
   elements, its first load reversed along every other axis from the first,
   whose axes so no two merge: of 8 axes it runs the direct kernel, of 9 the
   tape; both give nx.cpu's bits. *)
let test_direct_rank () =
  let b = backend () in
  let f32 = D.Float32 in
  let add r =
    let shape = Array.make r 2 in
    let n = 1 lsl r in
    let st = Random.State.make [| r |] in
    let h () =
      A.of_array f32 (Iarray.of_array shape)
        (Array.init n (fun _ -> Random.State.float st 2. -. 1.))
    in
    let x = h () and y = h () in
    let p =
      P.v ~ins:[| D.Any f32; D.Any f32 |]
        [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
        ~outs:[| 2 |]
    in
    let run (module K : Nx_kernel.S) dev =
      let on x = if Rig.equal dev Rig.host then x else A.to_device dev x in
      let flips =
        Array.init r (fun a ->
            if a mod 2 = 0 then { M.start = 1; count = 2; step = -1 }
            else { M.start = 0; count = 2; step = 1 })
      in
      let x' = Option.get (A.move (M.Slice (Iarray.of_array flips)) (on x)) in
      let z = A.create dev f32 (Iarray.of_array shape) in
      equal answer A.Done
        (K.map p ~dsts:[| A.Any z |]
           [| Nx_kernel.Plain x'; Nx_kernel.Plain (on y) |]);
      elements z
    in
    let want = run (module Nx_cpu) Rig.host in
    let got, traced = Plan.traced (fun () -> run (module Nx_metal) b.device) in
    equal (array string) want got;
    List.map
      (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
      (directs traced)
  in
  equal ~msg:"8 axes" (list string) [ direct_kernel ] (add 8);
  equal ~msg:"9 axes" (list string) [] (add 9)

(* nx_metal.mli: a map whose program leaves a load unread, of a form the direct
   kernel takes, runs the direct kernel with nx.cpu's bits: at 1024 elements,
   where it is flat, and at 1024 x 1024 with a load transposed, where it takes
   tiles. *)
let test_unread_load () =
  let b = backend () in
  let f32 = D.Float32 in
  let p =
    P.v
      ~ins:[| D.Any f32; D.Any f32; D.Any f32 |]
      [| P.In 0; In 1; In 2; Op2 (Binary Add, 0, 2) |]
      ~outs:[| 3 |]
  in
  let add shape =
    let n = Array.fold_left ( * ) 1 shape in
    let st = Random.State.make [| n |] in
    let h () =
      A.of_array f32 (Iarray.of_array shape)
        (Array.init n (fun _ -> Random.State.float st 2. -. 1.))
    in
    let x = h () and y = h () and u = h () in
    let run (module K : Nx_kernel.S) dev =
      let on x = if Rig.equal dev Rig.host then x else A.to_device dev x in
      let t x =
        if Array.length shape < 2 then x
        else Option.get (A.move (M.Permute (Iarray.of_list [ 1; 0 ])) x)
      in
      let z = A.create dev f32 (Iarray.of_array shape) in
      equal answer A.Done
        (K.map p ~dsts:[| A.Any z |]
           [| Nx_kernel.Plain (t (on x)); Plain (on u); Plain (on y) |]);
      elements z
    in
    let want = run (module Nx_cpu) Rig.host in
    let got, traced = Plan.traced (fun () -> run (module Nx_metal) b.device) in
    equal (array string) want got;
    List.map
      (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
      (directs traced)
  in
  equal ~msg:"1024" (list string) [ "loop_map_direct32_flat" ] (add [| 1024 |]);
  equal ~msg:"1024 x 1024" (list string)
    [ "loop_map_direct32_tiled" ]
    (add [| 1024; 1024 |])

(* nx_metal.mli: a cast whose load a map would stage takes the tape's tiles
   where their estimate is the lower. A float32 square of 1024 x 1024 cast
   transposed into float16 runs the tape's tiled map, and as it lies its cast's
   kernel, each with nx.cpu's bits. *)
let test_cast_transposed () =
  let b = backend () in
  let n = 1024 in
  let st = Random.State.make [| n |] in
  let x =
    A.of_array D.Float32
      (Iarray.of_list [ n; n ])
      (Array.init (n * n) (fun _ -> Random.State.float st 2. -. 1.))
  in
  let cast transposed =
    let run (module K : Nx_kernel.S) dev =
      let x = if Rig.equal dev Rig.host then x else A.to_device dev x in
      let x =
        if transposed then
          Option.get (A.move (M.Permute (Iarray.of_list [ 1; 0 ])) x)
        else x
      in
      let z = A.create dev D.Float16 (Iarray.of_list [ n; n ]) in
      equal answer A.Done (Support.map1 (module K) P.Cast ~dst:z x);
      elements z
    in
    let want = run (module Nx_cpu) Rig.host in
    let got, traced = Plan.traced (fun () -> run (module Nx_metal) b.device) in
    equal (array string) want got;
    List.map (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel) traced
  in
  equal ~msg:"transposed" (list string) [ "loop_map_tiled" ] (cast true);
  equal ~msg:"as it lies" (list string) [ "loop_cast_f32_16" ] (cast false)

(* nx_metal.mli: every float16 code cast to float32 runs its cast's kernel, with
   nx.cpu's bits: signs, zeros, subnormals, infinities and NaNs with their
   payloads. *)
let test_f16_codes () =
  let b = backend () in
  let n = 1 lsl 16 in
  let codes =
    A.of_array D.Uint16 (Iarray.of_array [| n |]) (Array.init n Fun.id)
  in
  let h = Option.get (A.bitcast D.Float16 codes) in
  let p =
    P.v ~ins:[| D.Any D.Float16 |]
      [| P.In 0; Op1 (Cast, D.Any D.Float32, 0) |]
      ~outs:[| 1 |]
  in
  let run (module K : Nx_kernel.S) dev =
    let x = if Rig.equal dev Rig.host then h else A.to_device dev h in
    let z = A.create dev D.Float32 (Iarray.of_array [| n |]) in
    equal answer A.Done (K.map p ~dsts:[| A.Any z |] [| Nx_kernel.Plain x |]);
    elements z
  in
  let want = run (module Nx_cpu) Rig.host in
  let got, traced = Plan.traced (fun () -> run (module Nx_metal) b.device) in
  equal (list string) [ "loop_cast_f16_32" ]
    (List.map (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel) traced);
  equal (array string) want got

(* nx_metal.mli: the tape has nx.cpu's bits at every float16 code, at 4 elements
   a thread and at 1, where an int64 operand rides beside: in a decode, an
   encode, a float16 Add, which decodes, adds and encodes, and a sum's store
   into float16. The encoded float32s are every high half beside low halves
   about float16's rounding bit, among them ties with the kept bit even and odd;
   the sum takes them but the subnormals, which Metal flushes in arithmetic
   (nx_metal.mli). An Add's or a sum's NaN follows the NaN rule (Prog's kinds),
   its payload unspecified, so any NaN stands for one. *)
let test_f16_tape () =
  let b = backend () in
  let n = 1 lsl 16 in
  let shape k = Iarray.of_array [| k |] in
  let f16 xs =
    let k = Array.length xs in
    A.Any (Option.get (A.bitcast D.Float16 (A.of_array D.Uint16 (shape k) xs)))
  in
  let codes = f16 (Array.init n Fun.id) in
  let turned = f16 (Array.init n (fun i -> (i + 12345) land 0xffff)) in
  let lows = [| 0; 0x1000; 0x1001; 0x2000; 0x2FFF; 0x3000 |] in
  let m = n * Array.length lows in
  let bits i =
    let l = Array.length lows in
    Int32.of_int (((i / l) lsl 16) lor lows.(i mod l))
  in
  let floats s bs =
    let s = Iarray.of_array s in
    A.Any (Option.get (A.bitcast D.Float32 (A.of_array D.Int32 s bs)))
  in
  let every = Array.init m bits in
  let normal =
    let sub b =
      Int32.logand b 0x7f800000l = 0l && Int32.logand b 0x7fffffl <> 0l
    in
    Array.of_list (List.filter (fun b -> not (sub b)) (Array.to_list every))
  in
  let on dev (A.Any x) =
    A.Any (if Rig.equal dev Rig.host then x else A.to_device dev x)
  in
  let plain (A.Any x) = Nx_kernel.Plain x in
  (* [node] of [xs], and an int64 operand beside where [beside], by [K] on
     [dev]: its result's elements. *)
  let map ~beside xs node (module K : Nx_kernel.S) dev =
    let k = List.length xs in
    let (A.Any x0) = List.hd xs in
    let numel = L.numel (A.layout x0) in
    let ins = List.map (fun (A.Any x) -> D.Any (A.dtype x)) xs in
    let ins = if beside then ins @ [ D.Any D.Int64 ] else ins in
    let nodes = List.init k (fun i -> P.In i) @ [ node ] in
    let nodes = if beside then nodes @ [ P.In k ] else nodes in
    let outs = if beside then [ k; k + 1 ] else [ k ] in
    let p =
      P.v ~ins:(Iarray.of_list ins) (Iarray.of_list nodes)
        ~outs:(Iarray.of_list outs)
    in
    let (D.Any dt) = P.dtype p k in
    let z = A.create dev dt (shape numel) in
    let ids =
      A.Any (A.of_array D.Int64 (shape numel) (Array.init numel Int64.of_int))
    in
    let loads = List.map plain (List.map (on dev) xs) in
    let loads = if beside then loads @ [ plain (on dev ids) ] else loads in
    let dsts = [ A.Any z ] in
    let dsts =
      if beside then dsts @ [ A.Any (A.create dev D.Int64 (shape numel)) ]
      else dsts
    in
    equal answer A.Done
      (K.map p ~dsts:(Iarray.of_list dsts) (Iarray.of_list loads));
    elements z
  in
  (* A sum of each float and a +0 along an axis, into float16: two terms, since
     a reduction of one term runs as a map. *)
  let sum (module K : Nx_kernel.S) dev =
    let k = Array.length normal in
    let s = [| k; 2 |] in
    let z = A.create dev D.Float16 (shape k) in
    let id = P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0 |] in
    let terms =
      Array.init (2 * k) (fun i -> if i mod 2 = 0 then normal.(i / 2) else 0l)
    in
    equal answer A.Done
      (K.reduce id ~shape:(Iarray.of_array s) ~axes:[| 1 |]
         [| (Monoid Sum, 0) |]
         ~dsts:[| A.Any z |]
         [| plain (on dev (floats s terms)) |]);
    elements z
  in
  let nan e =
    let h = Char.code e.[0] lor (Char.code e.[1] lsl 8) in
    if h land 0x7fff > 0x7c00 then "NaN" else e
  in
  let check ?(rule = Fun.id) what f kernel =
    let want = Array.map rule (f (module Nx_cpu : Nx_kernel.S) Rig.host) in
    let got, traced =
      Fun.protect
        ~finally:(fun () -> Nx_metal.bound_direct true)
        (fun () ->
          Nx_metal.bound_direct false;
          Plan.traced (fun () -> f (module Nx_metal : Nx_kernel.S) b.device))
    in
    let name (t : Plan.traced) = Nx_metal.Kernels.name t.kernel in
    equal ~msg:what (list string) [ kernel ] (List.map name traced);
    equal ~msg:what (array string) want (Array.map rule got)
  in
  let decode = P.Op1 (Cast, D.Any D.Float32, 0) in
  let encode = P.Op1 (Cast, D.Any D.Float16, 0) in
  let add = P.Op2 (Binary Add, 0, 1) in
  List.iter
    (fun (beside, kernel) ->
      let what s = Printf.sprintf "%s by %s" s kernel in
      check (what "decode") (map ~beside [ codes ] decode) kernel;
      let floats = floats [| m |] every in
      check (what "encode") (map ~beside [ floats ] encode) kernel;
      check ~rule:nan (what "add") (map ~beside [ codes; turned ] add) kernel)
    [ (false, "loop_map_v4"); (true, "loop_map_v1") ];
  check ~rule:nan "a sum's store" sum "loop_reduce_v4"
(* Casts by name *)

(* The dtypes a cast reads, all but those whose values only move, and those a
   cast's kernel writes: of a byte or more, but complex64. *)
let cast_sources =
  List.filter
    (fun dt -> dt <> D.Any D.Float64 && dt <> D.Any D.Complex128)
    D.all

let cast_targets =
  List.filter
    (fun (D.Any d as dt) -> D.bits d >= 8 && dt <> D.Any D.Complex64)
    cast_sources

let integer (D.Any d) = D.is D.Signed d || D.is D.Unsigned d

(* How nx_metal.mli runs a cast of [s] into [t]: as copies between integers of
   one width of a byte or more, whose bits it keeps; by its kernel, of [s]'s
   class and [t]'s width, into a target of [cast_targets]; else by the tape. The
   byte sources (uint8, int8, bool and the float8s) are one class and the nibble
   sources (uint4, int4, float4_e2m1fn) another; any other source is a class of
   its own. *)
type cast_route = Copies | Kernel of string | Tape

let cast_route (D.Any s as src) (D.Any t as dst) =
  let short (D.Any d) =
    match d with
    | D.Uint8 | D.Int8 | D.Bool | D.Float8_e4m3fn | D.Float8_e5m2 -> "byte"
    | D.Uint4 | D.Int4 | D.Float4_e2m1fn -> "nibble"
    | D.Float32 -> "f32"
    | D.Float16 -> "f16"
    | D.Bfloat16 -> "bf16"
    | D.Int64 -> "i64"
    | D.Uint64 -> "u64"
    | D.Int32 -> "i32"
    | D.Uint32 -> "u32"
    | D.Int16 -> "i16"
    | D.Uint16 -> "u16"
    | D.Complex64 -> "c64"
    | D.Bit -> "bit"
    | d -> D.name d
  in
  if src = dst then Tape
  else if integer src && integer dst && D.bits s = D.bits t && D.bits s >= 8
  then Copies
  else if List.mem dst cast_targets then
    Kernel (Printf.sprintf "loop_cast_%s_%d" (short src) (D.bits t))
  else Tape

(* An integer of [bits] bits' bounds and their neighbours; the edges past which
   float16, float8_e4m3fn and float8_e5m2 round to infinity or saturate, with a
   neighbour each: 65504, 65519 and 65520; 448, 464 and 465; 57344, 61439 and
   61440. The values that rounding through float32 would round twice into
   bfloat16: 2{^k} + 2{^(k-8)} + 1 lies past the half of bfloat16's last place,
   and float32 rounds it onto that half. And float32's ties from an integer it
   cannot hold, 2{^k} + 2{^(k-24)} and one past it for each k from 24, where a
   conversion in two steps rounds twice. *)
let int_edges ~signed bits =
  let k = if signed then bits - 2 else bits - 1 in
  let p e = Int64.shift_left 1L e in
  (* In 64 bits, 2^63 wraps to min_int and 2^63 - 1 to max_int. *)
  let lo = if signed then Int64.neg (p (bits - 1)) else 0L in
  let hi =
    if signed then Int64.pred (p (bits - 1))
    else if bits = 64 then -1L
    else Int64.pred (p bits)
  in
  let twice = Int64.(add (add (p k) (p (k - 8))) 1L) in
  let ties = [ twice; Int64.pred twice; Int64.sub twice 2L ] in
  let narrow =
    [ 65504L; 65519L; 65520L; 448L; 464L; 465L; 57344L; 61439L; 61440L ]
  in
  let f32_ties =
    List.concat_map
      (fun j ->
        let t = Int64.add (p j) (p (j - 24)) in
        [ t; Int64.succ t ])
      (List.init (k - 23) (fun i -> 24 + i))
  in
  let edges = ties @ narrow @ f32_ties in
  [
    lo;
    Int64.succ lo;
    Int64.pred hi;
    hi;
    0L;
    1L;
    Int64.add (p 24) 1L;
    Int64.add (p 24) 3L;
  ]
  @ edges
  @ if signed then List.map Int64.neg edges else []

(* float32's values at its targets' edges: - float16's and bfloat16's largest
   finite values, the last value that rounds to one and the first that rounds
   past it, float16's least subnormal and the ties around it, and bfloat16's
   ties to even; - float8_e4m3fn's 448, the tie at 464 below and above it, 465
   and 480; its least subnormal 2{^-9}, the tie 2{^-10} and past it; its ties to
   even 1.0625 and 1.1875; - float8_e5m2's 57344, 61440 and below it; 2{^-16},
   the tie 2{^-17} and past it; its ties 1.125 and 1.375; - each integer
   target's bounds, the largest float32 below them and the next past them:
   ±2{^31}, 2{^32}, ±2{^63}, 2{^64}, 127.5, 128, -128.5, -129, 255.5, 256,
   32767.5, -32768.5, 65535.5, 65536; - -1, -0.5 and the largest float32 below
   1. *)
let f32_edges =
  [
    0x477fe000L;
    0x477fefffL;
    0x477ff000L;
    0x33800000L;
    0x33000000L;
    0x33000001L;
    0x7f7fffffL;
    0x7f7f7fffL;
    0x7f7f8000L;
    0x00000001L;
    0x80000000L;
    0x3f808000L;
    0x3f818000L;
    0xbf808000L;
    0x43e00000L;
    0x43e7ffffL;
    0x43e80000L;
    0x43e88000L;
    0x43f00000L;
    0xc3e80000L;
    0x3b000000L;
    0x3a800000L;
    0x3a800001L;
    0x3f880000L;
    0x3f980000L;
    0x47600000L;
    0x47700000L;
    0x476fffffL;
    0xc7700000L;
    0x37800000L;
    0x37000000L;
    0x37000001L;
    0x3f900000L;
    0x3fb00000L;
    0x4f000000L;
    0x4effffffL;
    0xcf000000L;
    0xcf000001L;
    0x4f800000L;
    0x4f7fffffL;
    0x5f000000L;
    0x5effffffL;
    0xdf000000L;
    0xdf000001L;
    0x5f800000L;
    0x5f7fffffL;
    0x42ff0000L;
    0xc2ff0000L;
    0x43000000L;
    0xc3008000L;
    0xc3010000L;
    0x437f8000L;
    0x43800000L;
    0x46ffff00L;
    0xc7000080L;
    0x477fff80L;
    0x47800000L;
    0xbf800000L;
    0xbf000000L;
    0x3f7fffffL;
  ]

(* Every code of [dt], where it has at most 2{^16}; else its specials and edges.
   Each as its bytes. *)
let cast_codes (D.Any d as dt) =
  let w = D.bits d / 8 in
  let le x = Kernels_cases.le w x in
  if D.bits d <= 16 then
    List.init (1 lsl D.bits d) (fun c -> le (Int64.of_int c))
  else
    specials dt
    @
    match d with
    | D.Float32 -> List.map le f32_edges
    | D.Int64 | D.Int32 -> List.map le (int_edges ~signed:true (D.bits d))
    | D.Uint64 | D.Uint32 -> List.map le (int_edges ~signed:false (D.bits d))
    | D.Complex64 ->
        (* Real parts at float32's edges, imaginary parts 0 and 1. *)
        List.concat_map
          (fun re ->
            List.map
              (fun im -> le Int64.(logor re (shift_left im 32)))
              [ 0L; 0x3f800000L ])
          f32_edges
    | _ -> []

(* An array of [dt] and shape [s] on the GPU: its codes ({!cast_codes}) from the
   [from]th on, in turn, and drawn bytes past them. For a dtype narrower than a
   byte, every byte. *)
let cast_drawn =
  let codes = Hashtbl.create 16 in
  let codes dt =
    match Hashtbl.find_opt codes dt with
    | Some c -> c
    | None ->
        let c = Array.of_list (cast_codes dt) in
        Hashtbl.add codes dt c;
        c
  in
  fun st ~from (D.Any d as dt) s ->
    let x = A.create Rig.host d (Iarray.of_array s) in
    let b = A.buffer x in
    let n = Rig.Buffer.length b in
    let bytes = Bytes.create n in
    if D.bits d < 8 then
      for i = 0 to n - 1 do
        Bytes.unsafe_set bytes i (Char.unsafe_chr ((i + from) land 0xff))
      done
    else begin
      let codes = codes dt in
      let k = Array.length codes and w = D.bits d / 8 in
      for i = 0 to (n / w) - 1 do
        if i < k then
          Bytes.blit_string codes.((i + from) mod k) 0 bytes (i * w) w
        else
          for j = 0 to w - 1 do
            Bytes.unsafe_set bytes
              ((i * w) + j)
              (Char.chr (Random.State.int st 256))
          done
      done
    end;
    Rig.Buffer.blit_from_string (Bytes.unsafe_to_string bytes) 0 b 0 n;
    A.Any (A.to_device (S.device ()) x)

(* Whether [r], an element of [t] as its bytes, is a NaN, and a signalling one.
   float8_e4m3fn's one NaN of each sign has no quiet bit. *)
let nan_of (D.Any t) r =
  let v = ref 0 in
  for i = Int.min 7 (String.length r - 1) downto 0 do
    v := (!v lsl 8) lor Char.code r.[i]
  done;
  let v = !v in
  match t with
  | D.Float32 ->
      ( v land 0x7f800000 = 0x7f800000 && v land 0x7fffff <> 0,
        v land 0x400000 = 0 )
  | D.Float16 -> (v land 0x7c00 = 0x7c00 && v land 0x3ff <> 0, v land 0x200 = 0)
  | D.Bfloat16 -> (v land 0x7f80 = 0x7f80 && v land 0x7f <> 0, v land 0x40 = 0)
  | D.Float8_e4m3fn -> (v land 0x7f = 0x7f, false)
  | D.Float8_e5m2 -> (v land 0x7c = 0x7c && v land 0x3 <> 0, v land 0x2 = 0)
  | _ -> (false, false)

(* A cast's view of its operand, into a whole destination: a view of
   {!viewed}'s, at the shape its row takes; or [Off (a, b)], each contiguous,
   the operand from its buffer's element [a] and the destination from its [b]:
   off its vectors' size unless a multiple of 4; or [Transposed], a matrix's
   transpose, which a map would stage. *)
type cast_view = View of view | Off of int * int | Transposed

type cast_case = {
  src : D.any;
  dst : D.any;
  cview : cast_view;
  cshape : int array;
}

(* The path [c] was made to reach in its cast's kernel (direct.metal, Casts by
   name): flat quads where the operand and the destination are whole, or
   contiguous from elements on their vectors' size, over a positive multiple of
   4 elements, but from bit; quads where they are whole or contiguous over any
   other positive count, a walk of one axis whose last quad may be short, or
   over rows of 256; else lanes. A cast of no element launches nothing: Loop.map
   plans none ([Ok None]). *)
let cast_path c =
  let n = Array.fold_left ( * ) 1 c.cshape in
  let last = c.cshape.(Array.length c.cshape - 1) in
  let flat = c.src <> D.Any D.Bit in
  match c.cview with
  | _ when n = 0 -> "no cast's kernel"
  | View Whole -> if flat && n mod 4 = 0 then "flat quads" else "quads"
  | Off (a, b) ->
      if flat && n mod 4 = 0 && a mod 4 = 0 && b mod 4 = 0 then "flat quads"
      else "quads"
  | View (Rows _) when last mod 4 = 0 -> "quads"
  | View _ | Transposed -> "lanes"

let cast_id c =
  let name (D.Any d) = D.name d in
  Format.asprintf "%s into %s, %s over [%s]" (name c.src) (name c.dst)
    (match c.cview with
    | View v -> Format.asprintf "%a" pp_view v
    | Off (a, b) -> Format.asprintf "contiguous from %d into from %d" a b
    | Transposed -> "transposed")
    (String.concat "; " (Array.to_list (Array.map string_of_int c.cshape)))

(* Every source into every target, the operand whole at every code a shape holds
   (2{^16} codes for 16 bits). A kernel's views depend on its class and width
   alone, and the copies' on the width, so the first pair of [cast_sources] and
   [cast_targets] each kernel takes, and the first the copies take at each
   width, take every view besides, over rows of 257: of no element, of one and
   of 1024; its rows apart by 1 and by 3; every other element; each axis
   reversed; broadcast along each axis; and off by an element on either side or
   on both. Rows of 257 take lanes where the walk keeps two axes and quads where
   it is one, its last quad short; rows of 256 and 1024 quads, those of the
   array whole flat (direct.metal, Casts by name). Each view writes a
   destination whole or offset, the C-contiguous ones nx_kernel.mli asks of a
   caller. And the first source of each class into each target no kernel writes,
   which the tape casts. *)
let cast_cases =
  let views =
    [
      View (Rows 1);
      View (Rows 3);
      View Stepped;
      View Flipped;
      View Reversed;
      View (Spread 0);
      View (Spread 1);
      Off (1, 1);
    ]
  in
  let pairs =
    List.concat_map
      (fun src ->
        List.filter_map
          (fun dst -> if src = dst then None else Some (src, dst))
          cast_targets)
      cast_sources
  in
  (* The first pair of each kernel, and the first source of each class, by its
     kernel of width 8. *)
  let firsts key =
    List.fold_left
      (fun seen p ->
        match key p with
        | Some k when not (List.mem_assoc k seen) -> (k, p) :: seen
        | Some _ | None -> seen)
      [] pairs
    |> List.map snd
  in
  let kernel (src, dst) =
    match cast_route src dst with Kernel k -> Some k | Copies | Tape -> None
  in
  let copies (src, dst) =
    match (cast_route src dst, src) with
    | Copies, D.Any s -> Some ("copies of " ^ string_of_int (D.bits s))
    | (Kernel _ | Tape), _ -> None
  in
  let full = firsts kernel and copied = firsts copies in
  let class_of (src, dst) =
    match (kernel (src, dst), dst) with
    | Some k, D.Any t when D.bits t = 8 -> Some k
    | _ -> None
  in
  let declined =
    List.filter (fun dt -> not (List.mem dt cast_targets)) cast_sources
  in
  let tape =
    List.concat_map
      (fun (src, _) ->
        List.filter_map
          (fun dst ->
            if src = dst then None
            else Some { src; dst; cview = View Whole; cshape = [| 257 |] })
          declined)
      (firsts class_of)
  in
  List.concat_map
    (fun ((src, dst) as p) ->
      let (D.Any s) = src in
      let rows = if D.bits s = 16 then 256 else 5 in
      let case cview cshape = { src; dst; cview; cshape } in
      case (View Whole) [| rows; 257 |]
      ::
      (if not (List.mem p full || List.mem p copied) then []
       else
         List.map (fun v -> case v [| 5; 257 |]) views
         @ List.map (case (View Whole)) [ [| 0 |]; [| 1 |]; [| 1024 |] ]
         @ [
             case (Off (1, 1)) [| 1024 |];
             case (Off (1, 0)) [| 1024 |];
             case (Off (0, 1)) [| 1024 |];
             case (Off (4, 8)) [| 1024 |];
             case (View (Rows 3)) [| 4; 256 |];
             case Transposed [| 65; 67 |];
           ]))
    pairs
  @ tape

(* [c]'s cast on nx.metal and on nx.cpu, from operands drawn alike: the two
   destinations' arrays' elements, and nx.metal's launches. *)
let cast_run c =
  let st = Random.State.make [| Hashtbl.hash (cast_id c) |] in
  let s = c.cshape in
  let x, dbase, d =
    match c.cview with
    | View v ->
        let x, _ = viewed (fun dt s -> cast_drawn st ~from:3 dt s) c.src s v in
        let d = cast_drawn st ~from:0 c.dst s in
        (x, d, d)
    | Transposed ->
        let (A.Any t) = cast_drawn st ~from:3 c.src [| s.(1); s.(0) |] in
        let x = Option.get (A.move (M.Permute (Iarray.of_list [ 1; 0 ])) t) in
        let d = cast_drawn st ~from:0 c.dst s in
        (A.Any x, d, d)
    | Off (a, b) ->
        let n = Array.fold_left ( * ) 1 s in
        let shape = Iarray.of_array s in
        let from offset =
          L.v ~offset ~strides:(L.strides (L.contiguous shape)) shape
        in
        let (A.Any xb) = cast_drawn st ~from:5 c.src [| n + a |] in
        let x = A.Any (A.v (A.dtype xb) (from a) (A.buffer xb)) in
        let (A.Any db) = cast_drawn st ~from:0 c.dst [| n + b |] in
        (x, A.Any db, A.Any (A.v (A.dtype db) (from b) (A.buffer db)))
  in
  let (A.Any hb) = dbase in
  let hbase = A.Any (A.copy (Support.host hb)) in
  let hd =
    match c.cview with
    | View _ | Transposed -> hbase
    | Off _ ->
        let (A.Any h) = hbase in
        A.Any
          (A.v (A.dtype h)
             (let (A.Any dd) = d in
              A.layout dd)
             (A.buffer h))
  in
  let (A.Any xa) = x in
  let (A.Any da) = d in
  let (A.Any ha) = hd in
  let got, traced =
    Plan.traced (fun () -> Support.map1 (module Nx_metal) P.Cast ~dst:da xa)
  in
  equal answer A.Done got;
  equal answer A.Done
    (Support.map1 (module Nx_cpu) P.Cast ~dst:ha (Support.host xa));
  (Support.elements dbase, Support.elements hbase, traced)

let cast_names traced =
  List.map (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel) traced

let is_cast n = String.starts_with ~prefix:"loop_cast" n

let is_copy n =
  List.mem n
    [
      "copy";
      "copy_8";
      "copy_16";
      "copy_32";
      "copy_64";
      "copy_run";
      "copy_rows";
      "copy_rows_wide";
      "copy_step_8";
      "copy_step_16";
      "copy_step_32";
      "copy_step_64";
    ]

(* nx_metal.mli: a map of one plain load cast into a dtype of a byte or more but
   complex64 runs its kernel of the source's class and the target's width, a
   cast between integers of one width the copies, and every cast gives nx.cpu's
   bits but a NaN's sign and payload, quiet where nx.cpu's is. *)
let law_cast c =
  let got, want, traced = cast_run c in
  let names = cast_names traced in
  let casts = List.filter is_cast names in
  let mode (t : Plan.traced) =
    Int64.to_int (Bytes.get_int64_le t.words (8 * Nx_metal.Kernels.loop_mode))
  in
  let has bit =
    List.exists
      (fun (t : Plan.traced) ->
        is_cast (Nx_metal.Kernels.name t.kernel) && mode t land bit <> 0)
      traced
  in
  (match cast_route c.src c.dst with
  | Kernel k when c.cview = Transposed ->
      (* nx_metal.mli: its kernel, or the tape's tiles where their estimate is
         the lower. *)
      let ran = List.sort_uniq compare names in
      if ran <> [ "loop_map_tiled" ] then
        equal ~msg:"its cast's kernel or the tape's tiles" (list string) [ k ]
          ran
  | Kernel k ->
      let path =
        if casts = [] then "no cast's kernel"
        else if has Nx_metal.Kernels.loop_flat then "flat quads"
        else if has Nx_metal.Kernels.loop_quads then "quads"
        else "lanes"
      in
      equal ~msg:"its path, as its row was made to reach" string (cast_path c)
        path;
      let want = if path = "no cast's kernel" then [] else [ k ] in
      equal ~msg:"its cast's kernel" (list string) want
        (List.sort_uniq compare names)
  | Copies ->
      equal ~msg:"copies alone" (list string) []
        (List.filter (fun n -> not (is_copy n)) names)
  | Tape -> equal ~msg:"no cast's kernel" (list string) [] casts);
  (* Each element checked, its message made where it fails. *)
  Array.iteri
    (fun i w ->
      let g = got.(i) in
      let wn, ws = nan_of c.dst w and gn, gs = nan_of c.dst g in
      if wn then begin
        if not gn then
          equal ~msg:(Printf.sprintf "element %d is a NaN" i) bool true gn;
        if gs && not ws then
          equal ~msg:(Printf.sprintf "element %d signals" i) bool true ws
      end
      else if not (String.equal w g) then
        equal ~msg:(Printf.sprintf "element %d" i) string w g)
    want

(* nx_metal.mli: [bound_direct false] makes a cast run its tape, with the same
   bits, NaNs' included: over quads whose last is short and over flat quads,
   from every code of a table's sources into each width, and from wider
   sources. *)
let test_cast_bound () =
  List.iter
    (fun ((src, dst), cshape) ->
      let c = { src; dst; cview = View Whole; cshape } in
      let got, _, _ = cast_run c in
      let tape, _, traced =
        Fun.protect
          ~finally:(fun () -> Nx_metal.bound_direct true)
          (fun () ->
            Nx_metal.bound_direct false;
            cast_run c)
      in
      equal ~msg:"the tape alone" (list string) []
        (List.filter is_cast (cast_names traced));
      equal (array string) got tape)
    (List.concat_map
       (fun pair -> [ (pair, [| 3; 257 |]); (pair, [| 1024 |]) ])
       D.
         [
           (Any Float8_e4m3fn, Any Float16);
           (Any Float8_e4m3fn, Any Float32);
           (Any Float8_e5m2, Any Bfloat16);
           (Any Float8_e5m2, Any Float8_e4m3fn);
           (Any Float8_e4m3fn, Any Uint8);
           (Any Bool, Any Int64);
           (Any Int8, Any Float32);
           (Any Uint8, Any Int16);
           (Any Float4_e2m1fn, Any Float32);
           (Any Uint4, Any Uint8);
           (Any Int4, Any Float8_e5m2);
           (Any Bit, Any Int32);
           (Any Uint64, Any Bfloat16);
           (Any Float32, Any Bfloat16);
           (Any Float32, Any Int8);
           (Any Int64, Any Int32);
         ])

(* nx_metal.mli: a cast of uint4 into uint8 runs its kernel over flat quads and
   keeps each value, also past 2^30 elements, where a nibble's bit place passes
   2^32. Element [i] of the source is bits 28 to 31 of [i * 2654435761], which
   differs from element [i - 2^30]'s at bit 30. Every element past 2^30 is
   checked, the 4096 before it, the first 4096 and one every 2^20 + 1. It holds
   about 2 GiB, the source's 0.5 GiB on the host and the GPU and the
   destination's 1 GiB, and takes 2.2 s on an M1 Max. *)
let test_nibble_cast_far () =
  let b = backend () in
  let n = (1 lsl 30) + (1 lsl 16) in
  let nibble i = ((i * 2654435761) lsr 28) land 15 in
  let x = A.create b.device D.Uint4 (Iarray.of_list [ n ]) in
  let src = Bytes.create (n / 2) in
  for k = 0 to (n / 2) - 1 do
    let i = 2 * k in
    Bytes.unsafe_set src k
      (Char.unsafe_chr (nibble i lor (nibble (i + 1) lsl 4)))
  done;
  Rig.Buffer.blit_from_string
    (Bytes.unsafe_to_string src)
    0 (A.buffer x) 0 (Bytes.length src);
  let z = A.create b.device D.Uint8 (Iarray.of_list [ n ]) in
  let got, traced =
    Plan.traced (fun () -> Support.map1 (module Nx_metal) P.Cast ~dst:z x)
  in
  equal answer A.Done got;
  let flat (t : Plan.traced) =
    let mode = Bytes.get_int64_le t.words (8 * Nx_metal.Kernels.loop_mode) in
    Int64.to_int mode land Nx_metal.Kernels.loop_flat <> 0
  in
  equal ~msg:"its cast's kernel" (list string) [ "loop_cast_nibble_8" ]
    (List.sort_uniq compare (cast_names traced));
  equal ~msg:"flat quads" (list bool) [ true ]
    (List.sort_uniq compare (List.map flat traced));
  let check first count =
    let s = Bytes.create count in
    Rig.Buffer.blit_to_bytes (A.buffer z) first s 0 count;
    for k = 0 to count - 1 do
      let i = first + k in
      let g = Char.code (Bytes.get s k) in
      if g <> nibble i then
        equal ~msg:(Printf.sprintf "element %d" i) int (nibble i) g
    done
  in
  check 0 4096;
  check ((1 lsl 30) - 4096) (n - (1 lsl 30) + 4096);
  let step = (1 lsl 20) + 1 in
  for k = 0 to (n - 1) / step do
    check (k * step) 1
  done

(* nx_metal.mli: a map past a launch's buffers runs its outputs in passes, and a
   pass whose outputs each move an operand's bits copies them, as
   {!Nx_metal.moves} do. A sum of 11 int32 loads beside a move of a 12th load is
   14 operands and destinations: the sum takes a loop of its 12, the move the
   copies, with the operand's bits: 1024 contiguous elements on 16 bytes in one
   launch of copy_run, every other element from the second in one of copy_step
   of its width. *)
let test_split_moves () =
  let st = Random.State.make [| 5 |] in
  let n = 1024 in
  List.iter
    (fun ((D.Any s, moved), stepped) ->
      let xs = Array.init 11 (fun _ -> drawn st D.Int32 [| n |]) in
      let a =
        if not stepped then drawn st s [| n |]
        else
          let b = A.buffer (drawn st s [| 2 * n |]) in
          A.v s (L.v ~offset:1 ~strides:[| 2 |] [| n |]) b
      in
      let ins =
        Iarray.init 12 (fun i -> if i < 11 then D.Any D.Int32 else D.Any s)
      in
      (* Nodes 12 to 21 add the 11 loads in order. *)
      let sums =
        List.init 10 (fun i ->
            P.Op2 (Binary Add, (if i = 0 then 0 else 11 + i), i + 1))
      in
      let nodes = List.init 12 (fun i -> P.In i) @ sums @ [ moved 11 ] in
      let p = P.v ~ins (Iarray.of_list nodes) ~outs:[| 21; 22 |] in
      let (D.Any t) = P.dtype p 22 in
      let sum = A.create (S.device ()) D.Int32 [| n |] in
      let y = A.create (S.device ()) t [| n |] in
      let loads : Nx_kernel.load iarray =
        Iarray.of_list
          (List.map (fun x -> Nx_kernel.Plain x) (Array.to_list xs)
          @ [ Nx_kernel.Plain a ])
      in
      let got, launches =
        Plan.traced (fun () ->
            Nx_metal.map p ~dsts:[| A.Any sum; A.Any y |] loads)
      in
      equal answer A.Done got;
      let names =
        List.map
          (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
          launches
      in
      equal ~msg:"the move's launches" (list string)
        [
          (if stepped then Printf.sprintf "copy_step_%d" (D.bits s)
           else "copy_run");
        ]
        (List.filter (fun k -> is_copy k || is_cast k) names);
      equal ~msg:"the move's bits" (array string) (elements a) (elements y);
      let hsum = A.create Rig.host D.Int32 [| n |] in
      let hy = A.create Rig.host t [| n |] in
      let hloads : Nx_kernel.load iarray =
        Iarray.map
          (function
            | Nx_kernel.Plain x -> Nx_kernel.Plain (Support.host x) | l -> l)
          loads
      in
      equal answer A.Done (Nx_cpu.map p ~dsts:[| A.Any hsum; A.Any hy |] hloads);
      equal ~msg:"the sum" (array string) (elements hsum) (elements sum))
    (List.concat_map
       (fun m -> [ (m, false); (m, true) ])
       [
         (D.Any D.Int64, fun i -> P.Op1 (Cast, D.Any D.Uint64, i));
         (D.Any D.Uint32, fun i -> P.Op1 (Cast, D.Any D.Int32, i));
         (D.Any D.Int16, fun i -> P.Op1 (Bitcast, D.Any D.Float16, i));
         (D.Any D.Int64, fun i -> P.In i);
       ])

(* nx_kernel.mli: a map's destination may be strided, with negative strides;
   nx_metal.mli: moves copy into C-contiguous ones, and a loop computes the
   others. Each move, alone and beside a sum of 11 loads past a launch's
   buffers, into every other element from [n] of a buffer of [4 n], and into
   elements [n] to [2 n - 1] reversed, stores its operand's bits there and
   changes no other element of the buffer. *)
let test_moves_into_views () =
  let st = Random.State.make [| 7 |] in
  let n = 1024 in
  let views =
    [
      ("stepped", L.v ~offset:n ~strides:[| 2 |] [| n |]);
      ("reversed", L.v ~offset:((2 * n) - 1) ~strides:[| -1 |] [| n |]);
    ]
  in
  let moves =
    [
      ( "a cast of int64 into uint64",
        D.Any D.Int64,
        fun i -> P.Op1 (Cast, D.Any D.Uint64, i) );
      ( "a cast of uint32 into int32",
        D.Any D.Uint32,
        fun i -> P.Op1 (Cast, D.Any D.Int32, i) );
      ( "a bitcast of int16 into float16",
        D.Any D.Int16,
        fun i -> P.Op1 (Bitcast, D.Any D.Float16, i) );
      ( "a copy of float32",
        D.Any D.Float32,
        fun i -> P.Op1 (Copy, D.Any D.Float32, i) );
      ("an int8 load", D.Any D.Int8, fun i -> P.In i);
    ]
  in
  (* The program of the move of load [k] after [k] int32 loads summed. *)
  let program (D.Any s) moved k =
    let ins =
      Iarray.init (k + 1) (fun i -> if i < k then D.Any D.Int32 else D.Any s)
    in
    let sums =
      List.init
        (Int.max 0 (k - 1))
        (fun i -> P.Op2 (Binary Add, (if i = 0 then 0 else k + i), i + 1))
    in
    let nodes = List.init (k + 1) (fun i -> P.In i) @ sums @ [ moved k ] in
    let last = List.length nodes - 1 in
    let outs = if k = 0 then [ last ] else [ last - 1; last ] in
    let outs = Iarray.of_list outs in
    P.v ~ins (Iarray.of_list nodes) ~outs
  in
  List.iter
    (fun (move, (D.Any s as ds), moved) ->
      List.iter
        (fun k ->
          List.iter
            (fun (view, l) ->
              let msg =
                Printf.sprintf "%s%s, into a %s destination" move
                  (if k = 0 then "" else " beside a sum")
                  view
              in
              let p = program ds moved k in
              let last = Iarray.get p.outs (Iarray.length p.outs - 1) in
              let (D.Any t) = P.dtype p last in
              let a = drawn st s [| n |] in
              let xs =
                List.init k (fun _ ->
                    Nx_kernel.Plain (drawn st D.Int32 [| n |]))
              in
              let base = drawn st t [| 4 * n |] in
              let was = elements base in
              let y = A.v t l (A.buffer base) in
              let dsts =
                if k = 0 then [ A.Any y ]
                else
                  let sum = A.create (S.device ()) D.Int32 [| n |] in
                  [ A.Any sum; A.Any y ]
              in
              let dsts = Iarray.of_list dsts in
              let loads = Iarray.of_list (xs @ [ Nx_kernel.Plain a ]) in
              equal ~msg answer A.Done (Nx_metal.map p ~dsts loads);
              equal ~msg (array string) (elements a) (elements y);
              let reached = Array.make (4 * n) false in
              for i = 0 to n - 1 do
                reached.(L.offset l + (i * L.stride l 0)) <- true
              done;
              let got = elements base in
              let kept =
                List.filter
                  (fun i -> not reached.(i))
                  (List.init (4 * n) Fun.id)
              in
              equal
                ~msg:(msg ^ ": the rest of its buffer")
                (list string)
                (List.map (fun i -> was.(i)) kept)
                (List.map (fun i -> got.(i)) kept))
            views)
        [ 0; 11 ])
    moves

(* plan.mli: a call's launches past a part's seconds take several parts, the
   split map's copies among its launches. The map of test_split_moves, an int64
   load moved beside a sum of 11 int32 loads, is planned; then, under a part's
   bound halfway into its copies' time, in two parts or more; and under one of
   all its launches' time, in one. *)
let test_split_budget () =
  let st = Random.State.make [| 6 |] in
  let n = 1 lsl 16 in
  let xs = Array.init 11 (fun _ -> drawn st D.Int32 [| n |]) in
  let a = drawn st D.Int64 [| n |] in
  let ins =
    Iarray.init 12 (fun i -> if i < 11 then D.Any D.Int32 else D.Any D.Int64)
  in
  let sums =
    List.init 10 (fun i ->
        P.Op2 (Binary Add, (if i = 0 then 0 else 11 + i), i + 1))
  in
  let nodes =
    List.init 12 (fun i -> P.In i) @ sums @ [ P.Op1 (Cast, D.Any D.Uint64, 11) ]
  in
  let p = P.v ~ins (Iarray.of_list nodes) ~outs:[| 21; 22 |] in
  let dsts : A.any iarray =
    [|
      A.Any (A.create (S.device ()) D.Int32 [| n |]);
      A.Any (A.create (S.device ()) D.Uint64 [| n |]);
    |]
  in
  let loads : Nx_kernel.load iarray =
    Iarray.of_list
      (List.map (fun x -> Nx_kernel.Plain x) (Array.to_list xs)
      @ [ Nx_kernel.Plain a ])
  in
  let plan () = loop_map ~holds:(fun _ -> 256) p ~dsts loads in
  let time (l : Plan.launch) =
    Plan.seconds
      ~threads:(Nx_metal.Kernels.threads l.kernel)
      ~groups:l.groups ~resident:l.resident l.cost
  in
  let copy (l : Plan.launch) = is_copy (Nx_metal.Kernels.name l.kernel) in
  let launches =
    match plan () with
    | Ok (Some (q : Plan.t)) -> Array.to_list q.launches
    | Ok None -> failf "no launch"
    | Error e -> failf "answered %a" Nx_array_support.pp_answer e
  in
  let sum ls = List.fold_left (fun t l -> t +. time l) 0. ls in
  let all = sum launches and copies = sum (List.filter copy launches) in
  greater ~msg:"the copies' time" float_exact ~than:0. copies;
  let under budget =
    S.parted ~seconds:budget (fun () ->
        match plan () with
        | Ok (Some (q : Plan.t)) ->
            Printf.sprintf "%d parts" (Int.min 2 (Array.length q.parts))
        | Ok None -> "no launch"
        | Error e -> Format.asprintf "%a" Nx_array_support.pp_answer e)
  in
  equal ~msg:"under all but half the copies' time" string "2 parts"
    (under (all -. (copies /. 2.)));
  equal ~msg:"under all the launches' time" string "1 parts"
    (under (all *. (1. +. 1e-9)))

(* [x]'s elements in a fresh buffer on the GPU from its byte [at]. *)
let at_byte at x =
  let src = A.buffer x in
  let n = Rig.Buffer.length src in
  let dst =
    Rig.Buffer.view
      (Rig.Buffer.create (S.device ()) (at + n))
      ~first:at ~length:n
  in
  Rig.Buffer.copy ~src ~dst;
  A.v (A.dtype x) (A.layout x) dst

(* nx_metal.mli: a call of a kept call's launches takes its submission at any
   base its dtype admits, and the direct kernel and the tape read and write its
   operands there as at a multiple of 16: maps of two loads into a destination,
   all 16 bytes or 2, 4, 8 or 12 bytes past it, over rows of 8 elements 9 or 12
   apart and over a contiguous run, of 8, 16 and 32 bits, by the direct kernels
   (the 32-bit one, a narrow float's or the one of bytes, where its threads take
   quads, the 32-bit flat one over the contiguous run) and by the tape. The
   first call of each kernel keeps one submission, which every other takes. *)
let test_kept_bases () =
  let d = S.device () in
  let st = Random.State.make [| 23 |] in
  let layouts =
    [ ([| 5; 8 |], Rows 1); ([| 5; 8 |], Rows 4); ([| 3; 4; 12 |], Whole) ]
  in
  let names ts =
    List.map (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel) ts
  in
  let kept ~direct (D.Any dt as ad) (s, view) =
    let base dt s = special_drawn st dt s in
    let A.Any x, _ = viewed base ad s view in
    let A.Any y, _ = viewed base ad s view in
    let A.Any z, _ = viewed base ad s view in
    let p =
      P.v ~ins:[| ad; ad |]
        [| P.In 0; In 1; Op2 (Binary Maximum, 0, 1) |]
        ~outs:[| 2 |]
    in
    let kernel = if direct then direct_kernel else "loop_map_v4" in
    (* A direct map of 32 bits takes the 32-bit kernel where its threads take
       quads, its flat kernel where its operands are contiguous, one of a narrow
       float its codec's, and one of bytes the kernel of bytes. *)
    let kernel_of traced =
      let quads (t : Plan.traced) =
        mode_of t land Nx_metal.Kernels.loop_quads <> 0
      in
      match traced with
      | [ t ] when direct && quads t && D.bits dt = 32 ->
          if view = Whole then "loop_map_direct32_flat" else "loop_map_direct32"
      | [ t ] when direct && quads t && ad = D.Any D.Bfloat16 ->
          "loop_map_direct16_bf16"
      | [ t ] when direct && quads t && ad = D.Any D.Float16 ->
          "loop_map_direct16_f16"
      | [ t ] when direct && quads t && D.bits dt = 8 -> "loop_map_direct8"
      | _ -> kernel
    in
    let call at =
      let z = at_byte at z in
      let loads : Nx_kernel.load iarray =
        [| Plain (at_byte at x); Plain (at_byte at y) |]
      in
      let a, traced =
        Fun.protect
          ~finally:(fun () -> Nx_metal.bound_direct true)
          (fun () ->
            Nx_metal.bound_direct direct;
            Plan.traced (fun () -> Nx_metal.map p ~dsts:[| A.Any z |] loads))
      in
      let kernel = kernel_of traced in
      let msg = Printf.sprintf "%s from byte %d by %s" (D.name dt) at kernel in
      equal ~msg answer A.Done a;
      equal ~msg (list string) [ kernel ] (names traced);
      (msg, elements z)
    in
    (* A quad is one vector where its 4 elements lie on 4 times their size, else
       4 elements. At float32, the whole layout's quads and the rows of 12's lie
       on 16 bytes from byte 16 and on none from byte 20, so those two calls
       take one branch each; the rows of 9 take both in one call. *)
    let bases =
      List.filter (fun at -> at mod D.bytes dt 1 = 0) [ 16; 18; 20; 24; 28 ]
    in
    match List.map call bases with
    | [] -> ()
    | (_, want) :: rest ->
        List.iter (fun (msg, got) -> equal ~msg (array string) want got) rest
  in
  let before = Nx_metal.kept_calls d in
  List.iter
    (fun direct ->
      List.iter
        (fun dt -> List.iter (kept ~direct dt) layouts)
        D.[ Any Uint8; Any Bfloat16; Any Int16; Any Float32 ])
    [ true; false ];
  equal ~msg:"kept" int (before + 6) (Nx_metal.kept_calls d)

(* Direct reductions *)

(* The direct reductions' kernels: the generic one, and those of a row's
   reduction and dtype (nx_metal.mli, Time). *)
let is_reduce_direct name =
  name = "loop_reduce_direct"
  || name = "loop_reduce_centered"
  || String.starts_with ~prefix:"loop_row_" name

(* The kernels of a reduction's groups: the tape's and the direct ones. *)
let is_reduce name =
  String.starts_with ~prefix:"loop_reduce_" name
  || String.starts_with ~prefix:"loop_row_" name

let reduce_directs traced =
  List.filter
    (fun (t : Plan.traced) -> is_reduce_direct (Nx_metal.Kernels.name t.kernel))
    traced

let direct_reductions =
  Nx_kernel.
    [ Monoid Sum; Monoid Max; Monoid Min; Arg Max; Arg Min; Logsumexp; Moments ]

let reduction_name : Nx_kernel.reduction -> string = function
  | Monoid Sum -> "Sum"
  | Monoid Prod -> "Prod"
  | Monoid Max -> "Max"
  | Monoid Min -> "Min"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"
  | Logsumexp -> "Logsumexp"
  | Moments -> "Moments"

let is_arg : Nx_kernel.reduction -> bool = function
  | Arg _ -> true
  | Monoid _ | Logsumexp | Moments -> false

(* How a reduction's elements are drawn: as {!special_drawn} draws them; from a
   few values, so that an extreme recurs ({!tied_drawn}); or each one of the
   given elements' bytes. *)
type values = Drawn | Tied | Among of string list

(* A reduction of one load of [rdt], over [view] of an array of shape [rshape],
   along [raxes], its elements drawn as [values]. *)
type reduced = {
  rdt : D.any;
  red : Nx_kernel.reduction;
  rshape : int array;
  raxes : int array;
  rview : view;
  values : values;
  rseed : int;
}

let pp_values ppf = function
  | Drawn -> ()
  | Tied -> Format.fprintf ppf ", tied"
  | Among vs ->
      let hex v =
        String.concat ""
          (List.rev_map
             (fun c -> Printf.sprintf "%02x" (Char.code c))
             (List.of_seq (String.to_seq v)))
      in
      Format.fprintf ppf ", among %s" (String.concat " " (List.map hex vs))

let pp_reduced ppf c =
  let (D.Any d) = c.rdt in
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "%s at %s over [%s] along [%s], %a%a, seed %d"
    (reduction_name c.red) (D.name d) (ints c.rshape) (ints c.raxes) pp_view
    c.rview pp_values c.values c.rseed

let reduced_terms c = Array.fold_left (fun n a -> n * c.rshape.(a)) 1 c.raxes

let reduced_outputs c =
  let n = ref 1 in
  Array.iteri
    (fun a e -> if not (Array.mem a c.raxes) then n := !n * e)
    c.rshape;
  !n

(* Shapes of up to 3 axes: of no element, of outputs of no term, one term or one
   output, rows of at most 8 terms, which share a slot, and outputs past 16
   blocks of 1024 terms, which take several groups; and the reduced axes: the
   last alone, the first alone, all or any. *)
let reduced =
  let open Gen in
  let extent =
    of_list ~pp:Format.pp_print_int
      [ 1; 2; 3; 4; 5; 7; 8; 9; 16; 33; 64; 257; 1000; 1025 ]
  in
  let drawn =
    let* r = int_range 1 3 in
    such_that (fun s -> numel s <= 1 lsl 15) (array ~size:(constant r) extent)
  in
  let shape =
    frequency
      [
        ( 2,
          of_list
            [
              [| 0 |];
              [| 3; 0 |];
              [| 0; 3 |];
              [| 2; 0; 3 |];
              [| 1 |];
              [| 1; 1 |];
              [| 5; 1 |];
              [| 1; 7 |];
            ] );
        (1, of_list [ [| 1 lsl 16 |]; [| 2; 20000 |]; [| 17000; 2 |] ]);
        (6, drawn);
      ]
  in
  let axes s =
    let r = Array.length s in
    let any =
      map
        (fun bits ->
          Array.of_list
            (List.filter
               (fun a -> bits land (1 lsl a) <> 0)
               (List.init r Fun.id)))
        (int_range 0 ((1 lsl r) - 1))
    in
    frequency
      [
        (3, constant [| r - 1 |]);
        (2, constant [| 0 |]);
        (1, constant (Array.init r Fun.id));
        (2, any);
      ]
  in
  let view =
    frequency
      [
        (4, constant Whole);
        (2, map (fun p -> Rows p) (int_range 1 3));
        (1, constant Stepped);
        (1, constant Flipped);
        (2, map (fun a -> Spread a) (int_range 0 2));
      ]
  in
  with_pp pp_reduced
    (let* rdt = of_list ~pp:Kernels_cases.pp_dtype direct_dtypes in
     let (D.Any d) = rdt in
     let* red =
       of_list (List.filter (fun r -> Nx_kernel.accepts r d) direct_reductions)
     in
     let* rshape = shape in
     let+ raxes = axes rshape
     and+ rview = view
     and+ values = of_list [ Drawn; Tied ]
     and+ rseed = nat in
     { rdt; red; rshape; raxes; rview; values; rseed })

(* A fresh array of [dt] and shape [s] on the GPU, each element one of 1 to 3
   values drawn as {!special_drawn} draws an element, or, in a quarter of the
   draws, one of the bytes of +0 and -0, so that an extreme ties the two
   zeros. *)
let tied_drawn st (D.Any d as dt) s =
  let x = A.create (S.device ()) d (Iarray.of_array s) in
  let sp = Array.of_list (specials dt) in
  let w = D.bytes d 1 in
  let one () =
    if Array.length sp > 0 && Random.State.bool st then
      sp.(Random.State.int st (Array.length sp))
    else random_bytes st w
  in
  let values =
    if Random.State.int st 4 = 0 then
      [| String.make w '\000'; String.make (w - 1) '\000' ^ "\x80" |]
    else Array.init (1 + Random.State.int st 3) (fun _ -> one ())
  in
  let n = numel s in
  let bytes =
    String.concat ""
      (List.init n (fun _ -> values.(Random.State.int st (Array.length values))))
  in
  Rig.Buffer.blit_from_string bytes 0 (A.buffer x) 0 (String.length bytes);
  A.Any x

(* A fresh array of [d] and shape [s] on the GPU, each element one of the
   elements' bytes [vs], drawn from [st]. *)
let among_drawn st vs (D.Any d) s =
  let x = A.create (S.device ()) d (Iarray.of_array s) in
  let vs = Array.of_list vs in
  let bytes =
    String.concat ""
      (List.init (numel s) (fun _ -> vs.(Random.State.int st (Array.length vs))))
  in
  Rig.Buffer.blit_from_string bytes 0 (A.buffer x) 0 (String.length bytes);
  A.Any x

(* [c]'s reduction, the direct kernel taken where [direct], from an operand
   drawn from [c]'s seed: its answer, its destinations' elements, the launches
   planned and the operand's elements in C order. *)
let reduced_run c ~direct =
  let st = Random.State.make [| c.rseed |] in
  let base dt s =
    match c.values with
    | Drawn -> special_drawn st dt s
    | Tied -> tied_drawn st dt s
    | Among vs -> among_drawn st vs dt s
  in
  let A.Any x, _ = viewed base c.rdt c.rshape c.rview in
  let shape = Iarray.of_array c.rshape in
  let out_shape =
    Array.of_list
      (List.filteri
         (fun a _ -> not (Array.mem a c.raxes))
         (Array.to_list c.rshape))
  in
  let dsts =
    Array.map
      (fun (D.Any d) ->
        A.Any (A.create (S.device ()) d (Iarray.of_array out_shape)))
      (match c.red with
      | Arg _ -> [| c.rdt; D.Any D.Int64 |]
      | Moments -> [| c.rdt; c.rdt |]
      | Monoid _ | Logsumexp -> [| c.rdt |])
  in
  let p = P.v ~ins:[| c.rdt |] [| P.In 0 |] ~outs:[| 0 |] in
  let answer, traced =
    Fun.protect
      ~finally:(fun () -> Nx_metal.bound_direct true)
      (fun () ->
        Nx_metal.bound_direct direct;
        Plan.traced (fun () ->
            Nx_metal.reduce p ~shape ~axes:(Iarray.of_array c.raxes)
              [| (c.red, 0) |]
              ~dsts:(Iarray.of_array dsts) [| Nx_kernel.Plain x |]))
  in
  (answer, Array.map Support.elements dsts, traced, Support.elements (A.Any x))

(* A reduction of one load by Sum, Max, Min or Arg, and a Logsumexp or Moments,
   whose first pass is such a reduction, give the tape's answer and bits, NaN
   payloads and an Arg's first position among ties included, and [bound_direct
   false] makes them run the tape. *)
let law_reduce_direct c =
  let (D.Any d) = c.rdt in
  (* A Max, Min or Arg of outputs of no term is refused by its shape: such a
     case is a Sum's, or a reduction along no axis at booleans, which no Sum
     takes. *)
  let refused =
    match c.red with
    | Monoid (Max | Min) | Arg _ -> true
    | Monoid (Sum | Prod) | Logsumexp | Moments -> false
  in
  let c =
    if reduced_terms c = 0 && reduced_outputs c > 0 && refused then
      if Nx_kernel.accepts (Monoid Sum) d then { c with red = Monoid Sum }
      else { c with raxes = [||] }
    else c
  in
  let a, want, tape, _ = reduced_run c ~direct:false in
  let a', got, direct, _ = reduced_run c ~direct:true in
  equal ~msg:"the tape alone" (list string) []
    (List.map
       (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
       (reduce_directs tape));
  equal answer a a';
  equal (array (array string)) want got;
  let names =
    List.map
      (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
      (reduce_directs direct)
  in
  (* [took]: the generic kernel, loop_reduce_direct; [row]: a row's; [second]: a
     Logsumexp's or Moments' second pass's, generic or a row's, alone where it
     takes its first pass in its launch ([once]). *)
  let took = List.mem "loop_reduce_direct" names in
  let row = List.exists (String.starts_with ~prefix:"loop_row_") names in
  let second =
    List.exists
      (fun n ->
        n = "loop_reduce_centered"
        || n = "loop_row_logsumexp_f32"
        || n = "loop_row_moments_f32")
      names
  in
  let once = second && List.for_all (fun n -> n = List.hd names) names in
  let terms = reduced_terms c and outputs = reduced_outputs c in
  let r = Array.length c.rshape in
  let groups =
    List.fold_left
      (fun n (t : Plan.traced) ->
        let g, _, _ = t.groups in
        n + g)
      0 (reduce_directs direct)
  in
  let float = List.mem c.rdt D.[ Any Float32; Any Bfloat16; Any Float16 ] in
  cover "the generic direct kernel" took;
  cover "a row's kernel" row;
  List.iter
    (fun red ->
      match red with
      | Nx_kernel.Logsumexp | Moments ->
          cover
            ("a direct " ^ reduction_name red ^ " in one launch")
            (once && c.red = red);
          cover
            ("a direct " ^ reduction_name red ^ " in two launches")
            (second && (not once) && c.red = red)
      | Monoid _ | Arg _ ->
          cover
            ("the generic direct kernel's " ^ reduction_name red)
            (took && c.red = red))
    direct_reductions;
  cover "loop_reduce_centered" (List.mem "loop_reduce_centered" names);
  cover "a direct second pass in one launch of an output's several blocks"
    (once && terms > 1024);
  cover "a direct second pass in one launch of outputs of 2 to 8 terms"
    (once && terms >= 2 && terms <= 8);
  cover "a direct second pass in one launch of several outputs a group"
    (once && outputs > 1 && terms <= 8 * 1024);
  cover "the generic direct kernel at a float" (took && float);
  cover "the generic direct kernel at a narrow float"
    (took && (D.Any d = D.Any D.Bfloat16 || D.Any d = D.Any D.Float16));
  cover "the generic direct kernel at a narrow integer"
    (took && D.bits d < 32 && (D.is D.Signed d || D.is D.Unsigned d));
  cover "the generic direct kernel at booleans" (took && D.Any d = D.Any D.Bool);
  cover "the generic direct kernel along the last axis alone"
    (took && c.raxes = [| r - 1 |] && r > 1);
  cover "the generic direct kernel along axes before the last"
    (took && r > 1 && not (Array.mem (r - 1) c.raxes));
  cover "the generic direct kernel over several axes"
    (took && Array.length c.raxes > 1);
  cover "the generic direct kernel of one term an output" (took && terms = 1);
  cover "the generic direct kernel of outputs of 2 to 8 terms"
    (took && terms >= 2 && terms <= 8);
  cover "the generic direct kernel of an output's several groups"
    (took && terms > 16 * 1024);
  cover "the generic direct kernel of several groups' outputs"
    (took && groups > 1);
  cover "the generic direct kernel of an Arg over tied values"
    (took && is_arg c.red && c.values = Tied && terms > 1);
  cover "the generic direct kernel over rows apart"
    (took && match c.rview with Rows _ -> true | _ -> false);
  cover "the generic direct kernel over a broadcast load"
    (took && match c.rview with Spread _ -> true | _ -> false);
  cover "the generic direct kernel over a reversed load"
    (took && c.rview = Flipped);
  cover "the generic direct kernel over a stepped load"
    (took && c.rview = Stepped);
  cover "outputs of no term" (outputs > 0 && terms = 0);
  cover "no element" (numel c.rshape = 0)

(* One-term reductions *)

(* A reduction of one load by a monoid, each output of one term: of [odt] over
   [oshape] along [oaxes], its term the load or, where [doubled], the load plus
   itself, its elements drawn from [oseed] as {!host_specials} draws them. The
   dtypes that read a subnormal as a zero come twice as often. *)
type one_term = {
  odt : D.any;
  monoid : Nx_kernel.monoid;
  oshape : int array;
  oaxes : int array;
  doubled : bool;
  oseed : int;
}

let pp_one_term ppf c =
  let (D.Any d) = c.odt in
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "%s%s at %s over [%s] along [%s], seed %d"
    (reduction_name (Monoid c.monoid))
    (if c.doubled then " of x + x" else "")
    (D.name d) (ints c.oshape) (ints c.oaxes) c.oseed

let one_term =
  let open Gen in
  with_pp pp_one_term
    (let pp = Kernels_cases.pp_dtype in
     let* odt =
       frequency
         [
           (2, of_list ~pp D.[ Any Float32; Any Bfloat16 ]);
           (1, of_list ~pp direct_dtypes);
           (1, of_list ~pp one_term_dtypes);
         ]
     in
     let (D.Any d) = odt in
     let* monoid =
       of_list
         (List.filter
            (fun m -> Nx_kernel.accepts (Monoid m) d)
            ([ Sum; Prod; Max; Min ] : Nx_kernel.monoid list))
     in
     let* n =
       of_list ~pp:Format.pp_print_int [ 1; 2; 3; 4; 5; 8; 1000; 4097 ]
     in
     let* oshape, oaxes =
       of_list
         [
           ([| n; 1 |], [| 1 |]);
           ([| 1; n |], [| 0 |]);
           ([| n |], [||]);
           ([| 2; 1; n |], [| 1 |]);
         ]
     in
     let* doubled =
       if P.accepts2 (Binary Add) d then bool else constant false
     in
     let+ oseed = nat in
     { odt; monoid; oshape; oaxes; doubled; oseed })

(* A Sum and a Prod of 4097 float32 terms, among which specials draw a
   subnormal, a -0 and a NaN, which draws reach in 4 to 5 % of the cases, so
   that each run reaches them. *)
let one_term_examples =
  List.map
    (fun (monoid, oseed) ->
      {
        odt = D.Any D.Float32;
        monoid;
        oshape = [| 4097; 1 |];
        oaxes = [| 1 |];
        doubled = false;
        oseed;
      })
    Nx_kernel.[ (Sum, 1); (Prod, 2) ]

(* nx_kernel.mli: an output of one term is that term folded from its monoid's
   start: [+0] plus it for a Sum, which makes [-0] a [+0]; the term itself for a
   Prod, a Max or a Min. So it has nx.cpu's bits, a NaN any NaN, where a Sum's
   subnormal term, and a load that the term adds to itself, is read as
   nx_metal.mli says Add reads it. *)
let law_one_term c =
  let b = backend () in
  let (D.Any d) = c.odt in
  let st = Random.State.make [| c.oseed |] in
  let h = A.expect d (host_specials st c.odt c.oshape) in
  let out =
    Array.of_list
      (List.filteri
         (fun a _ -> not (Array.mem a c.oaxes))
         (Array.to_list c.oshape))
  in
  let p =
    if c.doubled then
      P.v ~ins:[| c.odt |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |]
    else P.v ~ins:[| c.odt |] [| P.In 0 |] ~outs:[| 0 |]
  in
  let run (module K : Nx_kernel.S) dev x =
    let z = A.create dev d (Iarray.of_array out) in
    equal answer A.Done
      (K.reduce p ~shape:(Iarray.of_array c.oshape)
         ~axes:(Iarray.of_array c.oaxes)
         [| (Nx_kernel.Monoid c.monoid, 0) |]
         ~dsts:[| A.Any z |] [| Nx_kernel.Plain x |]);
    elements z
  in
  let sum, prod, extreme =
    match c.monoid with
    | Sum -> (true, false, false)
    | Prod -> (false, true, false)
    | Max | Min -> (false, false, true)
  in
  let read =
    if (sum || c.doubled) && Kernels_cases.flushes b "add" c.odt then
      A.expect d (Kernels_cases.flushed_view (A.Any h))
    else h
  in
  let want = run (module Nx_cpu) Rig.host read in
  let got = run (module Nx_metal) b.device (A.to_device b.device h) in
  equal (array (Elements.same c.odt)) want got;
  (* nx_metal.mli: such a reduction runs as a map of its terms. *)
  let kernels =
    let x = A.v d (L.contiguous (Iarray.of_array c.oshape)) (A.buffer h) in
    let z = A.v d (L.contiguous (Iarray.of_array out)) (A.buffer h) in
    match
      loop_reduce
        ~holds:(fun _ -> 1024)
        p ~shape:(Iarray.of_array c.oshape) ~axes:(Iarray.of_array c.oaxes)
        [| (Nx_kernel.Monoid c.monoid, 0) |]
        ~dsts:[| A.Any z |] [| Nx_kernel.Plain x |]
    with
    | Ok (Some pl) ->
        List.sort_uniq compare
          (Array.to_list
             (Array.map
                (fun (l : Plan.launch) ->
                  let n = Nx_metal.Kernels.name l.kernel in
                  if
                    String.starts_with ~prefix:"loop_map" n
                    || String.starts_with ~prefix:"loop_cast" n
                  then "a map's"
                  else n)
                pl.launches))
    | Ok None -> [ "no plan" ]
    | Error _ -> [ "declined" ]
  in
  equal (list string) [ "a map's" ] kernels;
  let float = List.mem c.odt D.[ Any Float32; Any Bfloat16 ] in
  let w = Kernels_cases.part_bytes c.odt in
  let bits = Kernels_cases.bits_of h in
  let parts =
    if not float then []
    else List.init (Array.length bits / w) (Kernels_cases.part_at bits w)
  in
  let exp = if w = 2 then 0x7f80 else 0x7f80_0000 in
  let man = if w = 2 then 0x7f else 0x7f_ffff in
  let nan v = v land exp = exp && v land man <> 0 in
  let subnormal = List.exists (Kernels_cases.subnormal w) parts in
  cover "a Sum's subnormal term" (sum && subnormal);
  cover "a Prod's subnormal term" (prod && subnormal);
  cover "a Sum's term of -0" (sum && List.mem (Kernels_cases.sign_bit w) parts);
  cover "a Sum's NaN term" (sum && List.exists nan parts);
  cover "a Max's or a Min's NaN term" (extreme && List.exists nan parts);
  cover "a Sum" sum;
  cover "a Prod" prod;
  cover "a Max or a Min" extreme;
  cover "a term past a load" c.doubled;
  cover "a dtype of no direct map" (not (List.mem c.odt direct_dtypes));
  cover "one output" (numel out = 1);
  cover "outputs of a short last quad" (numel out > 4 && numel out mod 4 <> 0);
  cover "outputs past a group's" (numel out > 1024)

(* A scan of one load by a monoid along an axis of one term, [carried] or not,
   its term the load or, where [doubled], the load plus itself, its load's
   elements and its carry's drawn from [cseed]. *)
type one_scan = {
  cdt : D.any;
  cmonoid : Nx_kernel.monoid;
  cshape : int array;
  caxis : int;
  carried : bool;
  cdoubled : bool;
  cseed : int;
}

let pp_one_scan ppf c =
  let (D.Any d) = c.cdt in
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "%s scan%s%s at %s over [%s] along %d, seed %d"
    (reduction_name (Monoid c.cmonoid))
    (if c.cdoubled then " of x + x" else "")
    (if c.carried then " from a carry" else "")
    (D.name d) (ints c.cshape) c.caxis c.cseed

let one_scan =
  let open Gen in
  with_pp pp_one_scan
    (let pp = Kernels_cases.pp_dtype in
     let* cdt =
       frequency
         [
           (2, of_list ~pp D.[ Any Float32; Any Bfloat16 ]);
           (1, of_list ~pp direct_dtypes);
           (1, of_list ~pp one_term_dtypes);
           (1, of_list ~pp D.[ Any Bool; Any Bit ]);
         ]
     in
     let (D.Any d) = cdt in
     let* cmonoid =
       of_list
         (List.filter
            (fun m -> Nx_kernel.accepts (Monoid m) d)
            ([ Sum; Prod; Max; Min ] : Nx_kernel.monoid list))
     in
     let* n =
       of_list ~pp:Format.pp_print_int [ 1; 2; 3; 4; 5; 8; 1000; 4097 ]
     in
     let* cshape, caxis =
       of_list [ ([| n; 1 |], 1); ([| 1; n |], 0); ([| 2; 1; n |], 1) ]
     in
     (* A complex product's partial products may be subnormal where no operand
        or result is: Metal flushes them, nx.cpu keeps them. *)
     let* carried =
       if cdt = D.Any D.Complex64 && cmonoid = Prod then constant false
       else bool
     in
     let* cdoubled =
       if P.accepts2 (Binary Add) d then bool else constant false
     in
     let+ cseed = nat in
     { cdt; cmonoid; cshape; caxis; carried; cdoubled; cseed })

(* nx_kernel.mli: a scan's result of one term is that term folded from the
   chain's start: from its carry, the carry first, else as a reduction's output
   of one term. So it has nx.cpu's bits, a NaN any NaN, where a subnormal
   operand or result of the fold, or a load the term adds to itself, is read or
   written as nx_metal.mli says the fold's kind does. A boolean carry is true
   wherever its byte is not 0, on both backends. *)
let law_one_term_scan c =
  let b = backend () in
  let (D.Any d) = c.cdt in
  let st = Random.State.make [| c.cseed |] in
  let h = A.expect d (host_specials st c.cdt c.cshape) in
  let carrier = Nx_kernel.carrier c.cdt in
  let (D.Any cd) = carrier in
  let carry_shape =
    Array.mapi (fun i e -> if i = c.caxis then 1 else e) c.cshape
  in
  let carry = A.expect cd (host_specials st carrier carry_shape) in
  let kind : Nx_kernel.monoid -> string = function
    | Sum -> "add"
    | Prod -> "mul"
    | Max -> "maximum"
    | Min -> "minimum"
  in
  (* The fold's kind runs where the scan has a carry, or for a Sum. A float16
     term is decoded into a float32 of no subnormal. *)
  let fold = c.carried || c.cmonoid = Sum in
  let flushes dt = fold && Kernels_cases.flushes b (kind c.cmonoid) dt in
  let flush flushes x =
    if flushes then A.expect (A.dtype x) (Kernels_cases.flushed_view (A.Any x))
    else x
  in
  let doubled = c.cdoubled && Kernels_cases.flushes b "add" c.cdt in
  let read = flush (flushes c.cdt || doubled) h in
  let carry_read = flush (flushes carrier) carry in
  let p =
    if c.cdoubled then
      P.v ~ins:[| c.cdt |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |]
    else P.v ~ins:[| c.cdt |] [| P.In 0 |] ~outs:[| 0 |]
  in
  let run (module K : Nx_kernel.S) dev x k =
    let z = A.create dev d (Iarray.of_array c.cshape) in
    let carry = if c.carried then Some (A.Any k) else None in
    equal answer A.Done
      (K.scan ?carry p ~axis:c.caxis (Monoid c.cmonoid, 0) ~dsts:[| A.Any z |]
         [| Nx_kernel.Plain x |]);
    z
  in
  let want =
    elements
      (flush (flushes c.cdt) (run (module Nx_cpu) Rig.host read carry_read))
  in
  let got, traced =
    Plan.traced (fun () ->
        run
          (module Nx_metal)
          b.device (A.to_device b.device h)
          (A.to_device b.device carry))
  in
  (* A boolean is true where its byte is not zero, whichever byte. *)
  let truths es =
    if c.cdt = D.Any D.Bool then
      Array.map (fun e -> if e = "\000" then e else "\001") es
    else es
  in
  equal (array (Elements.same c.cdt)) (truths want) (truths (elements got));
  (* nx_metal.mli: such a scan runs as a map of its terms. *)
  let map (t : Plan.traced) =
    let n = Nx_metal.Kernels.name t.kernel in
    String.starts_with ~prefix:"loop_map" n
    || String.starts_with ~prefix:"loop_cast" n
  in
  equal (list string) []
    (List.filter_map
       (fun t -> if map t then None else Some (Nx_metal.Kernels.name t.kernel))
       traced);
  let w = Kernels_cases.part_bytes c.cdt in
  let float = List.mem c.cdt D.[ Any Float32; Any Bfloat16 ] in
  let parts bits =
    if not float then []
    else List.init (Array.length bits / w) (Kernels_cases.part_at bits w)
  in
  let exp = if w = 2 then 0x7f80 else 0x7f80_0000 in
  let man = if w = 2 then 0x7f else 0x7f_ffff in
  let nan v = v land exp = exp && v land man <> 0 in
  let terms = parts (Kernels_cases.bits_of h) in
  cover "a Sum" (c.cmonoid = Sum);
  cover "a Prod" (c.cmonoid = Prod);
  cover "a Max or a Min" (c.cmonoid = Max || c.cmonoid = Min);
  cover "from a carry" c.carried;
  cover "a boolean carry of a byte past 1"
    (c.carried && carrier = D.Any D.Bool
    && Array.exists (fun b -> b > 1) (Kernels_cases.bits_of carry));
  cover "a term past a load" c.cdoubled;
  cover "a dtype of no direct map" (not (List.mem c.cdt direct_dtypes));
  cover "a NaN term from a carry" (c.carried && List.exists nan terms);
  cover "a subnormal term" (List.exists (Kernels_cases.subnormal w) terms);
  cover "results past a group's" (numel c.cshape > 1024)

(* Direct reductions along a row *)

let row_dtypes = D.[ Any Float32; Any Bfloat16; Any Float16 ]
let row_names = [ "sum"; "max"; "min"; "argmax"; "argmin" ]

let row_suffix (D.Any d) =
  if D.Any d = D.Any D.Float32 then "f32"
  else if D.Any d = D.Any D.Bfloat16 then "bf16"
  else if D.Any d = D.Any D.Float16 then "f16"
  else invalid_arg "row_suffix"

(* The kernel nx_metal.mli names for a direct reduction [red] of a [dt] load
   along a row, a Logsumexp's or Moments' first pass among them. *)
let row_kernel dt (red : Nx_kernel.reduction) =
  let r =
    match red with
    | Monoid Sum | Moments -> "sum"
    | Monoid Max | Logsumexp -> "max"
    | Monoid Min -> "min"
    | Arg Max -> "argmax"
    | Arg Min -> "argmin"
    | Monoid Prod -> invalid_arg "row_kernel"
  in
  Printf.sprintf "loop_row_%s_%s" r (row_suffix dt)

let row_kernels =
  [ "loop_row_logsumexp_f32"; "loop_row_moments_f32" ]
  @ List.concat_map
      (fun dt ->
        List.map
          (fun r -> Printf.sprintf "loop_row_%s_%s" r (row_suffix dt))
          row_names)
      row_dtypes

(* Whether [c]'s reduced axes of extent past 1, at least one, step through its
   operand [x] as one axis: each one's stride is the next one's times that one's
   extent (nx_metal.mli, Time). *)
let one_axis c (A.Any x) =
  let st = Iarray.to_array (L.strides (A.layout x)) in
  let rs = List.filter (fun a -> c.rshape.(a) > 1) (Array.to_list c.raxes) in
  let rec steps = function
    | a :: (b :: _ as rest) -> st.(a) = st.(b) * c.rshape.(b) && steps rest
    | [ _ ] | [] -> true
  in
  rs <> [] && steps rs

(* Reductions of a float32, bfloat16 or float16 load by each of Sum, Max, Min,
   Arg, Logsumexp and Moments. Where [along], the reduced axes are the last ones
   and the view keeps them one axis: whole or stepped past one reduced axis;
   along the last alone, any view {!reduced} draws, or that axis reversed or
   broadcast; else any axes and view, as {!reduced} draws them. *)
let rowed ~along =
  let open Gen in
  with_pp pp_reduced
  @@ let* rdt = of_list ~pp:Kernels_cases.pp_dtype row_dtypes in
     let (D.Any d) = rdt in
     let* red =
       of_list (List.filter (fun r -> Nx_kernel.accepts r d) direct_reductions)
     in
     let* c = reduced in
     let r = Array.length c.rshape in
     let+ raxes, rview =
       if not along then constant (c.raxes, c.rview)
       else
         let* k = int_range 0 (r - 1) in
         let raxes = Array.init (r - k) (fun i -> k + i) in
         let+ rview =
           if k = r - 1 then
             frequency
               [
                 (3, constant c.rview);
                 (1, constant Reversed);
                 (1, constant (Spread (r - 1)));
               ]
           else of_list [ Whole; Stepped ]
         in
         (raxes, rview)
     in
     { c with rdt; red; raxes; rview }

(* What a float element of [dt], as its bytes [b], holds. *)
type float_kind = Nan | Infinite | Zero | Subnormal | Normal

let float_kind (D.Any d) b =
  let w = D.bytes d 1 in
  let x = ref 0 in
  for i = w - 1 downto 0 do
    x := (!x lsl 8) lor Char.code b.[i]
  done;
  let exp, man =
    if w = 4 then (0x7f800000, 0x007fffff)
    else if D.Any d = D.Any D.Bfloat16 then (0x7f80, 0x007f)
    else (0x7c00, 0x03ff)
  in
  if !x land exp = exp then if !x land man <> 0 then Nan else Infinite
  else if !x land exp = 0 then if !x land man = 0 then Zero else Subnormal
  else Normal

let negative (D.Any d) b = Char.code b.[D.bytes d 1 - 1] land 0x80 <> 0

(* A reduction of a float load along a row by Sum, Max, Min or Arg, or a
   Logsumexp's or Moments' first pass, runs its reduction's and dtype's kernel,
   with the tape's answer and bits, a NaN's payload and an Arg's first position
   among ties included. *)
let law_row_direct c =
  let c =
    let (D.Any d) = c.rdt in
    if
      reduced_terms c = 0
      && reduced_outputs c > 0
      && match c.red with Monoid (Max | Min) | Arg _ -> true | _ -> false
    then
      if Nx_kernel.accepts (Monoid Sum) d then { c with red = Monoid Sum }
      else c
    else c
  in
  let a, want, _, _ = reduced_run c ~direct:false in
  let a', got, direct, xs = reduced_run c ~direct:true in
  equal answer a a';
  equal (array (array string)) want got;
  let ran =
    List.map
      (fun (t : Plan.traced) -> Nx_metal.Kernels.name t.kernel)
      (reduce_directs direct)
  in
  let terms = reduced_terms c and outputs = reduced_outputs c in
  let r = Array.length c.rshape in
  let kind = float_kind c.rdt in
  (* Whether some output's terms [ts] and value [v] satisfy [f]: the reduced
     axes are the last, so an output's terms lie together in C order. *)
  let some_output f =
    Array.length want > 0
    && terms > 0
    && List.exists
         (fun i -> f (Array.sub xs (i * terms) terms) want.(0).(i))
         (List.init outputs Fun.id)
  in
  let has ts f = Array.exists f ts in
  let extreme =
    match c.red with
    | Monoid (Max | Min) | Arg _ -> true
    | Monoid (Sum | Prod) | Logsumexp | Moments -> false
  in
  List.iter (fun k -> cover k (List.mem k ran)) row_kernels;
  let row = List.exists (String.starts_with ~prefix:"loop_row_") ran in
  cover "a row's kernel with a NaN result"
    (row && some_output (fun _ v -> kind v = Nan));
  cover "a row's kernel with a zero result"
    (row && extreme && some_output (fun _ v -> kind v = Zero));
  cover "a row's kernel's extreme over -0 and +0"
    (row && extreme
    && some_output (fun ts v ->
        kind v = Zero
        && has ts (fun t -> kind t = Zero && negative c.rdt t)
        && has ts (fun t -> kind t = Zero && not (negative c.rdt t))));
  cover "a row's kernel's subnormal extreme"
    (row && extreme && some_output (fun _ v -> kind v = Subnormal));
  cover "a row's kernel's infinite extreme"
    (row && extreme && some_output (fun _ v -> kind v = Infinite));
  cover "a row's kernel's Sum of an infinity and its negation"
    (row && c.red = Monoid Sum
    && some_output (fun ts _ ->
        has ts (fun t -> kind t = Infinite && negative c.rdt t)
        && has ts (fun t -> kind t = Infinite && not (negative c.rdt t))));
  cover "a row's kernel's Arg over a tie at the extreme"
    (row && is_arg c.red
    && some_output (fun ts v ->
        Array.fold_left (fun n t -> if t = v then n + 1 else n) 0 ts > 1));
  cover "outputs of no term" (outputs > 0 && terms = 0);
  cover "a row's kernel over a stepped load" (row && c.rview = Stepped);
  cover "a row's kernel over a reduced axis reversed"
    (row && (c.rview = Reversed || (c.rview = Flipped && Array.mem 0 c.raxes)));
  cover "a row's kernel over a reduced axis broadcast"
    (row
    &&
    match c.rview with
    | Spread a -> Array.mem (a mod r) c.raxes && c.rshape.(a mod r) > 1
    | _ -> false);
  cover "a row's kernel over rows apart"
    (row && match c.rview with Rows _ -> true | _ -> false);
  cover "a row's kernel over several axes" (row && Array.length c.raxes > 1);
  cover "a row's kernel of an output's several groups" (row && terms > 16 * 1024);
  cover "a row's kernel of outputs of 2 to 8 terms"
    (row && terms >= 2 && terms <= 8)

(* The bytes of the float32 [x]. *)
let f32_bytes x =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.bits_of_float x);
  Bytes.to_string b

(* An element of [d]'s bytes: the float32 [f32], or the bits [bf16] or [f16] of
   a 16-bit float. *)
let float_bytes (D.Any d) ~f32 ~bf16 ~f16 =
  if D.Any d = D.Any D.Float32 then f32_bytes f32
  else if D.Any d = D.Any D.Bfloat16 then le16 bf16
  else le16 f16

(* Each row kernel, over rows of 64 terms among few values, with an extreme that
   ties -0 and +0 (Min), a subnormal extreme (Max), a Sum of an infinity and its
   negation, an Arg tied at its extreme, an infinite extreme and rows apart. *)
let row_examples =
  let row ?(view = Whole) dt red vs =
    {
      rdt = dt;
      red;
      rshape = [| 8; 64 |];
      raxes = [| 1 |];
      rview = view;
      values = Among vs;
      rseed = 1;
    }
  in
  let v dt ~f32 ~bf16 ~f16 = float_bytes dt ~f32 ~bf16 ~f16 in
  let zeros dt =
    [
      v dt ~f32:0. ~bf16:0x0000 ~f16:0x0000;
      v dt ~f32:(-0.) ~bf16:0x8000 ~f16:0x8000;
      v dt ~f32:(Int32.float_of_bits 1l) ~bf16:0x0001 ~f16:0x0001;
    ]
  in
  let one dt = v dt ~f32:1. ~bf16:0x3f80 ~f16:0x3c00 in
  let two dt = v dt ~f32:2. ~bf16:0x4000 ~f16:0x4000 in
  let inf dt = v dt ~f32:Float.infinity ~bf16:0x7f80 ~f16:0x7c00 in
  let ninf dt = v dt ~f32:Float.neg_infinity ~bf16:0xff80 ~f16:0xfc00 in
  List.concat_map
    (fun dt ->
      Nx_kernel.
        [
          row dt (Monoid Min) (zeros dt);
          row dt (Monoid Max) (zeros dt);
          row dt (Monoid Sum) [ inf dt; ninf dt; one dt ];
          row dt (Arg Max) [ one dt; two dt ];
          row dt (Arg Min) [ one dt; two dt ];
        ])
    row_dtypes
  @ Nx_kernel.
      [
        row (D.Any D.Float32) (Monoid Max)
          [ inf (D.Any D.Float32); one (D.Any D.Float32) ];
        row ~view:(Rows 2) (D.Any D.Float32) (Monoid Max)
          [ one (D.Any D.Float32); two (D.Any D.Float32) ];
        (* A Logsumexp's and Moments' second pass by its float32 row kernel, and
           outputs of no term, 2 to 8 % of the draws. *)
        row (D.Any D.Float32) Logsumexp [ one (D.Any D.Float32) ];
        row (D.Any D.Float32) Moments [ one (D.Any D.Float32) ];
        {
          (row (D.Any D.Float32) (Monoid Sum) [ one (D.Any D.Float32) ]) with
          rshape = [| 3; 0 |];
        };
      ]

(* A Logsumexp, Moments and outputs of no term, by the direct kernels; and a
   Logsumexp's and Moments' second pass, in the launch of its first and after it
   (outputs past 16 blocks), at its edges: a NaN term in some lanes of an output
   only, NaNs throughout, an infinite maximum, whose center is 0, an infinite
   mean, every term -inf, and sums that overflow. *)
let reduced_examples =
  let c ?(values = Drawn) dt red rshape =
    { rdt = dt; red; rshape; raxes = [| 1 |]; rview = Whole; values; rseed = 1 }
  in
  let f32 = D.Any D.Float32 and bf16 = D.Any D.Bfloat16 in
  let among xs = Among (List.map f32_bytes xs) in
  let nan = Int32.float_of_bits 0x7fc01234l in
  (* One NaN among 32 values, so that an output of 64 terms holds about two, in
     its lanes' 16 at most two; among 20,000, about one an output of 20,000. *)
  let rare n =
    among (nan :: List.init (n - 1) (fun i -> Float.of_int i /. 8.))
  in
  Nx_kernel.
    [
      c f32 Logsumexp [| 4; 64 |];
      c bf16 Moments [| 4; 64 |];
      c f32 (Monoid Sum) [| 3; 0 |];
      c ~values:(rare 32) f32 Logsumexp [| 8; 64 |];
      c ~values:(rare 32) f32 Moments [| 8; 64 |];
      c ~values:(among [ 1.; nan; 2. ]) f32 Moments [| 4; 64 |];
      c ~values:(among [ Float.infinity; 1.; 100. ]) f32 Logsumexp [| 4; 64 |];
      c ~values:(among [ Float.neg_infinity ]) f32 Logsumexp [| 4; 64 |];
      c ~values:(among [ Float.infinity; 1.; 2. ]) f32 Moments [| 4; 64 |];
      c ~values:(among [ 3e38; -3e38; 1e38 ]) f32 Moments [| 4; 64 |];
      c f32 Logsumexp [| 2; 5000 |];
      c ~values:Tied bf16 Moments [| 3; 3000 |];
      c f32 Moments [| 64; 3 |];
      c ~values:(rare 20000) f32 Logsumexp [| 2; 20000 |];
      c ~values:(rare 20000) f32 Moments [| 2; 20000 |];
      c ~values:(among [ 3e38; -3e38; 1e38 ]) f32 Moments [| 2; 20000 |];
      c bf16 Logsumexp [| 2; 20000 |];
    ]
  (* The generic kernel at a float and a narrow float, along two axes apart, and
     at booleans, along axes before the last, over a reversed and a stepped
     load, of an output's several groups: 4 to 6 % of the draws each. *)
  @ Nx_kernel.
      [
        { (c f32 (Monoid Sum) [| 8; 9; 7 |]) with raxes = [| 0; 2 |] };
        {
          (c bf16 (Monoid Max) [| 8; 9; 7 |]) with
          raxes = [| 0; 2 |];
          rview = Flipped;
        };
        {
          (c (D.Any D.Int32) (Monoid Min) [| 17000; 2 |]) with
          raxes = [| 0 |];
          rview = Stepped;
        };
        c (D.Any D.Bool) (Monoid Max) [| 9; 64 |];
      ]

(* nx_metal.mli, Time: a direct reduction of a float32, bfloat16 or float16
   load, its pass's reduced axes one axis of every operand of the pass, runs the
   kernel of its reduction and dtype, and any other direct reduction the generic
   one: alone, and in a pass with a reduction of a load of another view, drawn
   from [c]'s seed. *)
let law_row_rule c =
  let st = Random.State.make [| c.rseed |] in
  let draw view =
    fst (viewed (fun dt s -> special_drawn st dt s) c.rdt c.rshape view)
  in
  let x = draw c.rview in
  let y =
    draw
      (match Random.State.int st 5 with
      | 0 -> Whole
      | 1 -> Rows 1
      | 2 -> Stepped
      | 3 -> Reversed
      | _ -> Spread (Random.State.int st 3))
  in
  let out_shape =
    Array.of_list
      (List.filteri
         (fun a _ -> not (Array.mem a c.raxes))
         (Array.to_list c.rshape))
  in
  let per = Nx_metal.Loop.destinations c.red in
  let ran ~n loads =
    let p =
      P.v
        ~ins:(Iarray.init n (fun _ -> c.rdt))
        (Iarray.init n (fun i -> P.In i))
        ~outs:(Iarray.init n Fun.id)
    in
    let dsts =
      Iarray.init (n * per) (fun i ->
          let (D.Any e) =
            if i mod per = 1 && is_arg c.red then D.Any D.Int64 else c.rdt
          in
          A.Any (A.create (S.device ()) e (Iarray.of_array out_shape)))
    in
    match
      loop_reduce
        ~holds:(Nx_metal.holds (S.device ()))
        p ~shape:(Iarray.of_array c.rshape) ~axes:(Iarray.of_array c.raxes)
        (Iarray.init n (fun i -> (c.red, i)))
        ~dsts
        (Iarray.of_list (List.map (fun (A.Any x) -> Nx_kernel.Plain x) loads))
    with
    | Ok (Some plan) ->
        List.filter is_reduce_direct
          (List.map
             (fun (l : Plan.launch) -> Nx_metal.Kernels.name l.kernel)
             (Array.to_list plan.launches))
    | Ok None | Error _ -> []
  in
  (* A reduction's kernels in order: a Logsumexp's or Moments' second pass's
     alone where its outputs have at most 16 blocks, else its first pass's then
     it; each kernel's launches one or more. *)
  let wants one =
    let first = if one then row_kernel c.rdt c.red else "loop_reduce_direct" in
    match c.red with
    | Logsumexp | Moments ->
        let second =
          if one && c.rdt = D.Any D.Float32 then
            if c.red = Logsumexp then "loop_row_logsumexp_f32"
            else "loop_row_moments_f32"
          else "loop_reduce_centered"
        in
        if reduced_terms c <= 16 * 1024 then [ second ] else [ first; second ]
    | Monoid _ | Arg _ -> [ first ]
  in
  let rec runs = function
    | a :: (b :: _ as rest) -> if a = b then runs rest else a :: runs rest
    | l -> l
  in
  let alone = ran ~n:1 [ x ] in
  let one = one_axis c x in
  equal
    ~msg:(Format.asprintf "%a" pp_reduced c)
    (list string)
    (if alone = [] then [] else wants one)
    (runs alone);
  List.iter (fun k -> cover k (List.mem k alone)) row_kernels;
  cover "the generic kernel at a float load"
    (List.mem "loop_reduce_direct" alone);
  let both = one && one_axis c y in
  let paired = ran ~n:2 [ x; y ] in
  equal ~msg:"two reductions in one pass" (list string)
    (if paired = [] then [] else runs (wants both @ wants both))
    (runs paired);
  cover "two reductions in one pass by a row's kernel" (both && paired <> []);
  cover
    "two reductions in one pass by the generic kernel, one stepping as one axis"
    (one && (not both) && paired <> [])

(* Two domains copying at once *)

let domain_elements = 4_096

(* Two domains copying at once, each between arrays of its own, each get their
   operand's bits; swaps, which stage through the device's workspace, get each
   other's. An array's bits are compared whole, as one string, and the operands'
   bytes are drawn once, before the domains start. Each call makes its own
   arrays, so that it can be drawn on a branch: one that takes a made value is
   drawn only after a prefix that made one, which 4 programs can all lack. One
   command draws a copy or a swap: each program holds a command only with
   probability 3/4, so two commands left one out of all 4 programs about one run
   in 400, and that run checked the other alone. *)
let domains_copy_alike =
  let bytes seed =
    random_bytes (Random.State.make [| seed |]) (4 * domain_elements)
  in
  let want = bytes 5 and other = bytes 6 in
  let draw b =
    let x = A.create (S.device ()) D.Int32 [| domain_elements |] in
    Rig.Buffer.blit_from_string b 0 (A.buffer x) 0 (4 * domain_elements);
    x
  in
  let bits x =
    let s = Bytes.create (4 * domain_elements) in
    Rig.Buffer.blit_to_bytes (A.buffer x) 0 s 0 (4 * domain_elements);
    Bytes.unsafe_to_string s
  in
  let copy () =
    let x = draw want in
    let dst = A.create (S.device ()) D.Int32 [| domain_elements |] in
    equal answer A.Done (Support.map1 (module Nx_metal) P.Copy ~dst x);
    [| bits dst |]
  in
  (* A swap of two arrays of its own, staged through the device's workspace,
     which the other domain's swaps share. *)
  let swap () =
    let x = draw want and y = draw other in
    let ins : D.any iarray = [| D.Any D.Int32; D.Any D.Int32 |] in
    let p = P.v ~ins [| P.In 0; P.In 1 |] ~outs:[| 1; 0 |] in
    let dsts : A.any iarray = [| A.Any x; A.Any y |] in
    let loads : Nx_kernel.load iarray = [| Plain x; Plain y |] in
    equal answer A.Done (Nx_metal.map p ~dsts loads);
    [| bits x; bits y |]
  in
  let expected swaps =
    cover "a copy" (not swaps);
    cover "a swap" swaps;
    if swaps then [| other; want |] else [| want |]
  in
  stateful ~domains:2 ~count:4 "domains copying at once get one call's bits"
    [
      command "run"
        (Gen.bool @-> returns (array string))
        expected
        (fun swaps -> if swaps then swap () else copy ());
    ]

(* The time budget *)

let launch_zero ~groups ?shared ?resident cost () =
  let b = Plan.make () in
  Plan.launch b Nx_metal.Kernels.Zero ~groups ?shared ?resident ~cost
    (Plan.words Nx_metal.Kernels.Zero)
    [| { Plan.word = Nx_metal.Kernels.zero_dst; slot = 0; write = true } |]

(* plan.mli: a launch whose estimate exceeds the budget is refused: its bytes at
   the bandwidth; its operations at one a cycle on each lane it fills, 128 for
   one group of 256 threads; or its chain once for each wave of 7 cores times
   the groups a core holds, at most 3072 threads and 64 KiB of threadgroup
   memory a core. *)
let test_refused () =
  let past = Invalid_argument "Plan.launch: past the time budget" in
  (* The counts that fill the budget past a launch's dispatch. *)
  let at rate = Float.to_int ((Plan.budget -. Plan.dispatch) *. rate) in
  let one = (1, 1, 1) in
  launch_zero ~groups:one (cost ~bytes:(at Plan.bandwidth) ()) ();
  raises past (launch_zero ~groups:one (cost ~bytes:(at Plan.bandwidth + 1) ()));
  let group = 128. *. 0.91 *. 1.278e9 in
  launch_zero ~groups:one (cost ~ops:(at group) ()) ();
  raises past (launch_zero ~groups:one (cost ~ops:(at group + 1) ()));
  let chain = at (0.6 *. Plan.lane_rate) in
  launch_zero ~groups:(7, 1, 1) (cost ~chain ()) ();
  raises past (launch_zero ~groups:(8, 1, 1) (cost ~chain ()));
  launch_zero ~groups:(84, 1, 1) ~resident:12 (cost ~chain ()) ();
  raises past (launch_zero ~groups:(85, 1, 1) ~resident:12 (cost ~chain ()));
  let resident = Invalid_argument "Plan.launch: resident" in
  raises resident (launch_zero ~groups:one ~resident:0 (cost ()));
  raises resident (launch_zero ~groups:one ~resident:13 (cost ()));
  launch_zero ~groups:one ~shared:(32 * 1024) ~resident:2 (cost ()) ();
  raises resident
    (launch_zero ~groups:one ~shared:(32 * 1024) ~resident:3 (cost ()));
  (* 21841 bytes, 21856 once rounded: two groups a core, not three. *)
  raises resident (launch_zero ~groups:one ~shared:21841 ~resident:3 (cost ()));
  raises (Invalid_argument "Plan.launch: cost")
    (launch_zero ~groups:one (cost ~ops:(-1) ()));
  raises (Invalid_argument "Nx_metal.bound_budget") (fun () ->
      Plan.bound_budget 0.);
  raises (Invalid_argument "Nx_metal.bound_budget") (fun () ->
      Plan.bound_budget (Plan.budget *. 2.))

(* plan.mli: a launch takes at most [most_refs] buffer refs, 13: the loop's 12
   buffers and its tail. *)
let test_most_refs () =
  let k = Nx_metal.Kernels.Loop_map_v1 in
  let launch n () =
    let ref i = { Plan.word = i; slot = i; write = i = 0 } in
    let refs = Array.init n ref in
    Plan.launch (Plan.make ()) k ~groups:(1, 1, 1) ~cost:(cost ())
      (Plan.words k) refs
  in
  equal ~msg:"most refs" int 13 Plan.most_refs;
  launch 13 ();
  raises (Invalid_argument "Plan.launch: refs") (launch 14)

(* Launches of few lanes *)

type narrow = { groups : int; threads : int; per : int; links : int }

let pp_narrow ppf c =
  Format.fprintf ppf "%d groups of %d threads, %d operations and %d links each"
    c.groups c.threads c.per c.links

let narrow =
  let open Gen in
  with_pp pp_narrow
    (let+ groups =
       frequency [ (2, constant 1); (3, int_range 2 7); (2, int_range 8 64) ]
     and+ threads = of_list [ 1; 32; 64; 128; 256; 1024 ]
     and+ per = int_range 1 1_000_000
     and+ links = int_range 0 1000 in
     { groups; threads; per; links })

(* One group of 32 threads, which a draw reaches one case in 21. *)
let narrow_examples = [ { groups = 1; threads = 32; per = 1000; links = 10 } ]

(* plan.mli: an estimate is a launch's dispatch, which a launch of no work
   takes, and at least the least time its counts take on the M1. A lane runs at
   most one operation a cycle at 1.278 GHz, and a group runs on one core of 128
   lanes, of 7; so [groups] groups of [threads] threads, each of [per]
   independent operations, take at least [per] cycles a thread, and their work
   over at most a thread a lane, 128 lanes a core and 7 cores. A group's chain
   runs once for each wave of 7 groups, one a core. *)
let law_narrow c =
  cover "one group of 32 threads" (c.groups = 1 && c.threads = 32);
  cover "more groups than cores" (c.groups > 7);
  let ops = c.groups * c.threads * c.per in
  let got =
    Plan.seconds ~threads:c.threads ~groups:(c.groups, 1, 1)
      (cost ~ops ~chain:c.links ())
  in
  let cycles = 1.278e9 in
  let a_thread = Float.of_int c.per /. cycles in
  let busy = Int.min (Int.min c.groups 7 * 128) (c.groups * c.threads) in
  let all = Float.of_int ops /. (Float.of_int busy *. cycles) in
  let waves = (c.groups + 6) / 7 in
  let chain = Float.of_int (waves * c.links) /. Plan.lane_rate in
  equal ~msg:"a launch of no work" float_exact Plan.dispatch
    (Plan.seconds ~threads:c.threads ~groups:(c.groups, 1, 1) (cost ()));
  at_least ~msg:"a dispatch" float_exact ~than:Plan.dispatch got;
  at_least ~msg:"a thread's work" float_exact ~than:a_thread got;
  at_least ~msg:"the work over the lanes" float_exact ~than:all got;
  at_least ~msg:"the chain" float_exact ~than:chain got

(* The planning of a call does not touch its device, so plans of arrays on the
   host are the plans of the GPU's: C-contiguous destinations, the operands
   themselves where they share a buffer, of operands strided or broadcast, under
   budgets that split their launches and bounds on a call's that decline
   some. *)
type budgeted = {
  dt : int;
  outputs : int;
  shape : int array;
  strides : int array option;
  shared : bool;
  tickets : int;
  budget : float option;
  call : float option;
}

let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a))

let pp_budgeted ppf c =
  let (D.Any dt) = List.nth rotatable c.dt in
  let seconds = Option.fold ~none:"" ~some:(Printf.sprintf ", %g s") in
  Format.fprintf ppf "%s, %d outputs of [%s]%s, %d tickets%s%s" (D.name dt)
    c.outputs (ints c.shape)
    (match (c.shared, c.strides) with
    | true, _ -> " sharing a buffer"
    | false, None -> " broadcast"
    | false, Some s -> Printf.sprintf " strides [%s]" (ints s))
    c.tickets (seconds c.budget)
    (Option.fold ~none:"" ~some:(Printf.sprintf ", a call %g s") c.call)

(* A bound on a call's seconds, half the time: from a tenth of [s] to 10 times
   it. *)
let call_bound s =
  let open Gen in
  frequency
    [
      (1, constant None);
      (1, map Option.some (float_range (s /. 10.) (10. *. s)));
    ]

(* Strided views of up to 4 axes of at most 16, of 4 axes half the time, whose
   copies past about 9000 elements take several launches, under budgets just
   past what one group of a copy of 4 axes takes. *)
let strided_case =
  let open Gen in
  let* r = frequency [ (1, int_range 0 3); (1, constant 4) ] in
  let* budget = float_range 3e-5 1e-4 in
  let+ dt = int_range 0 (List.length rotatable - 1)
  and+ outputs = int_range 1 3
  and+ shape = array ~size:(constant r) (int_range 0 16)
  and+ strides = array ~size:(constant r) (int_range (-9) 9)
  and+ shared = bool
  and+ tickets = frequency [ (2, constant 0); (1, int_range 1 4096) ]
  and+ call = call_bound budget in
  {
    dt;
    outputs;
    shape;
    strides = (if shared then None else Some strides);
    shared;
    tickets;
    budget = Some budget;
    call;
  }

(* Broadcast operands of up to 4 axes of at most 2^16 elements, some of no
   element, under the library's budget and bounds on a call's of a microsecond
   and more. *)
let broadcast_case =
  let open Gen in
  let extent =
    frequency
      [ (1, constant 0); (9, int_range 1 300); (3, int_range 1 (1 lsl 12)) ]
  in
  let shape =
    let* r = int_range 0 4 in
    map
      (fun s ->
        let n = Array.fold_left ( * ) 1 s in
        if n <= 1 lsl 16 then s else [| 1 lsl 16 |])
      (array ~size:(constant r) extent)
  in
  let+ dt = int_range 0 (List.length rotatable - 1)
  and+ outputs = int_range 1 3
  and+ shape = shape
  and+ shared = bool
  and+ tickets =
    frequency [ (2, constant 0); (1, int_range 1 (kept_tickets / 2)) ]
  and+ call = call_bound 1e-6 in
  { dt; outputs; shape; strides = None; shared; tickets; budget = None; call }

let budgeted =
  Gen.with_pp pp_budgeted
    (Gen.frequency [ (1, strided_case); (1, broadcast_case) ])

(* Copies of 16^4 strided elements under a launch's budget of 3e-5 s, which take
   several launches an output: of one output; of three that share a buffer,
   staged; and of one under a part's bound of that one launch's budget, which
   cuts it into parts. *)
let budget_examples =
  let c ?(outputs = 1) ?(shared = false) ?call () =
    {
      dt = 0;
      outputs;
      shape = [| 16; 16; 16; 16 |];
      strides = (if shared then None else Some [| 9; -7; 5; 3 |]);
      shared;
      tickets = 0;
      budget = Some 3e-5;
      call;
    }
  in
  [ c (); c ~outputs:3 ~shared:true (); c ~call:3e-5 () ]

(* An array of [dt] and [shape] over one element of [b]. *)
let broadcast dt shape b =
  A.v dt (L.v ~strides:(Iarray.map (fun _ -> 0) shape) shape) b

(* An array of [dt], [shape] and [strides] over a host buffer of its span. *)
let strided dt shape strides =
  let offset = ref 0 in
  Array.iteri
    (fun i s ->
      if s < 0 && shape.(i) > 0 then offset := !offset - ((shape.(i) - 1) * s))
    strides;
  let strides = Iarray.of_array strides in
  let l = L.v ~offset:!offset ~strides (Iarray.of_array shape) in
  let _, hi = L.span l in
  let b = Rig.Buffer.create Rig.host (Int.max 16 (hi * D.bits dt / 8)) in
  A.v dt l b

(* The calls nx.metal plans for [c] under a bound of [call] on a call's seconds,
   [c.call]'s by default: a rotation of its outputs where they share a buffer,
   copies into C-contiguous destinations of their own otherwise, or [Error ()]
   where the moves decline it; and a launch that takes [c.tickets] tickets, of
   their own, which launches put first zero. *)
let plans ?call c =
  let (D.Any dt) = List.nth rotatable c.dt in
  let shape = Iarray.of_array c.shape in
  let k = c.outputs in
  let operand () =
    match c.strides with
    | Some s -> strided dt c.shape s
    | None -> broadcast dt shape (Rig.Buffer.create Rig.host 16)
  in
  let own () = A.create Rig.host dt shape in
  let dsts = Array.init k (fun _ -> own ()) in
  let srcs = if c.shared then dsts else Array.init k (fun _ -> operand ()) in
  let p =
    P.v
      ~ins:(Iarray.init k (fun _ -> D.Any dt))
      (Iarray.init k (fun i -> P.In i))
      ~outs:(Iarray.init k (fun i -> (i + 1) mod k))
  in
  let moves () =
    match
      moves_map p
        ~dsts:(Iarray.map (fun x -> A.Any x) (Iarray.of_array dsts))
        (Iarray.map (fun x -> Nx_kernel.Plain x) (Iarray.of_array srcs))
    with
    | Ok None -> Ok []
    | Ok (Some c) -> Ok [ c ]
    | Error A.Declined -> Error ()
    | Error _ -> failwith "the moves refused a copy"
  in
  let tickets () =
    if c.tickets = 0 then []
    else begin
      let b = Plan.make () in
      let at = Plan.tickets b c.tickets in
      let w = Plan.words Nx_metal.Kernels.Zero in
      Plan.set w Nx_metal.Kernels.zero_dst at;
      Plan.set w Nx_metal.Kernels.zero_count 1;
      Plan.launch b Nx_metal.Kernels.Zero ~groups:(1, 1, 1)
        ~cost:{ Plan.bytes = 4; ops = 0; chain = 0 }
        w
        [|
          {
            Plan.word = Nx_metal.Kernels.zero_dst;
            slot = Plan.tickets_slot b;
            write = true;
          };
        |];
      [ all_fresh (fun () -> Plan.finish b) ]
    end
  in
  let plan () = (moves (), tickets ()) in
  let plan () =
    match Option.fold ~none:c.call ~some:Option.some call with
    | None -> plan ()
    | Some t -> S.parted ~seconds:t plan
  in
  match c.budget with None -> plan () | Some t -> bounded t plan

let seconds (l : Plan.launch) =
  Plan.seconds
    ~threads:(Nx_metal.Kernels.threads l.kernel)
    ~groups:l.groups ~resident:l.resident l.cost

let call_seconds (p : Plan.t) =
  Array.fold_left (fun s l -> s +. seconds l) 0. p.launches

(* plan.mli: nx.metal plans every call within the time budget, splitting it into
   launches, which the planner would otherwise refuse, and a call whose launches
   together pass a part's seconds into parts, each within them but a part of one
   launch. *)
let law_budget c =
  let moves, tickets = plans c in
  let moves =
    match moves with Ok ps -> ps | Error () -> fail "the moves declined"
  in
  let part = Option.value c.call ~default:Plan.part_budget in
  List.iter
    (fun (p : Plan.t) ->
      cover "a call of several parts" (Array.length p.parts > 1);
      Array.iter
        (fun (q : Plan.part) ->
          if Array.length q.run > 1 then
            at_most ~msg:"a part's estimate" float_exact ~than:part
              (Array.fold_left (fun t l -> t +. seconds l) 0. q.run))
        p.parts)
    moves;
  let n = Array.fold_left ( * ) 1 c.shape in
  let k = c.outputs in
  cover "no element" (n = 0);
  cover "a strided operand" (Option.is_some c.strides && n > 1);
  cover "a staged operand" (c.shared && k > 1 && n > 0);
  cover "tickets of their own" (c.tickets > 0);
  let launches (p : Plan.t) = Array.length p.launches in
  cover "an output in several launches"
    (List.exists (fun p -> launches p > unsplit k) moves
    || List.exists (fun p -> launches p > 2) tickets);
  let allowed = Option.value c.budget ~default:Plan.budget in
  List.iter
    (fun (p : Plan.t) ->
      Array.iter
        (fun (l : Plan.launch) ->
          at_most
            ~msg:(Nx_metal.Kernels.name l.kernel)
            float_exact ~than:allowed (seconds l))
        p.launches)
    (moves @ tickets)

(* A copy of 2^32 elements or more, of a dtype of a byte or more, planned over
   layouts of the workspace, which allocates nothing: within 2^32 more, or
   within 2^30 of 2^37, the most a GPU of the family addresses; from a source
   that steps by [step] elements, 0 for a broadcast one. *)
type huge = { dt : int; n : int; step : int }

let pp_huge ppf c =
  let (D.Any dt) = List.nth rotatable c.dt in
  Format.fprintf ppf "%s, %d elements stepping by %d" (D.name dt) c.n c.step

let huge =
  let open Gen in
  with_pp pp_huge
    (let+ dt = int_range 0 (List.length rotatable - 1)
     and+ n =
       frequency
         [
           (1, int_range (1 lsl 32) (1 lsl 33));
           (1, int_range ((1 lsl 37) - (1 lsl 30)) (1 lsl 37));
         ]
     and+ step = of_list ~pp:Format.pp_print_int [ 0; 1; 1; 2; -1 ] in
     { dt; n; step })

(* moves.mli: a copy's launches each keep within the time budget, as many as its
   elements take, and their seconds together are {!Moves.launch}'s result. *)
let law_huge c =
  let (D.Any dt) = List.nth rotatable c.dt in
  let n = c.n in
  let b = Plan.make () in
  let ws = Plan.workspace b in
  let site l =
    Nx_metal.Moves.site_at ~slot:ws ~base:0 ~residue:0 (D.Any dt) l
  in
  let offset = if c.step < 0 then (n - 1) * -c.step else 0 in
  let src =
    L.v ~offset ~strides:(Iarray.of_list [ c.step ]) (Iarray.of_list [ n ])
  in
  let dst = L.contiguous (Iarray.of_list [ n ]) in
  let t = Nx_metal.Moves.launch b ~n ~dst:(site dst) (site src) in
  let p = Plan.finish b in
  cover "2^32 elements or more, planned" (n >= 1 lsl 32);
  cover "an output in several launches" (Array.length p.launches > 1);
  cover "a broadcast source" (c.step = 0);
  cover "a source of consecutive elements" (c.step = 1);
  cover "a reversed source" (c.step < 0);
  Array.iter
    (fun (l : Plan.launch) ->
      at_most
        ~msg:(Nx_metal.Kernels.name l.kernel)
        float_exact ~than:(Plan.allowed ()) (seconds l))
    p.launches;
  equal ~msg:"the copy's seconds"
    (float_rel ~rel:1e-9 ~abs:0.)
    t (call_seconds p)

(* nx_metal.mli: a copy an element a thread reads a line of 128 bytes for each
   load of its unit of copy where its source's innermost axis steps 128 bytes or
   more: one load an element of an integer or a float, two of a complex number,
   stored on half its width. A transpose of 64 rows of 128 elements steps 128
   elements, and its launches may count at most two lines more a run of 64
   elements, and six more a launch. *)
let test_strided_reads () =
  let rows = 64 and cols = 128 in
  let n = rows * cols in
  List.iter
    (fun (D.Any dt, loads) ->
      let b = Plan.make () in
      let ws = Plan.workspace b in
      let site l =
        Nx_metal.Moves.site_at ~slot:ws ~base:0 ~residue:0 (D.Any dt) l
      in
      let src =
        L.v
          ~strides:(Iarray.of_list [ 1; cols ])
          (Iarray.of_list [ cols; rows ])
      in
      let dst = L.contiguous (Iarray.of_list [ cols; rows ]) in
      ignore (Nx_metal.Moves.launch b ~n ~dst:(site dst) (site src));
      let p = Plan.finish b in
      let charged =
        Array.fold_left
          (fun t (l : Plan.launch) -> t + l.cost.bytes)
          0 p.launches
      in
      let reads = n * ((D.bits dt / 8) + (loads * Plan.line)) in
      let ends = 6 * Plan.line * Array.length p.launches in
      at_least ~msg:(D.name dt) int ~than:reads charged;
      at_most ~msg:(D.name dt) int
        ~than:(reads + (2 * Plan.line * (n / rows)) + ends)
        charged)
    [
      (D.Any D.Uint8, 1);
      (D.Any D.Float16, 1);
      (D.Any D.Float32, 1);
      (D.Any D.Int64, 1);
      (D.Any D.Complex64, 2);
      (D.Any D.Complex128, 2);
    ]

(* nx_metal.mli: copy_step weighs each element it copies one at a time, in a
   block a row's end crosses or a partial block of the destination, 512
   operations more, and a block's elements more for as many groups of 32 blocks
   as the fewer of such blocks and groups, at least as many as the groups that
   hold one; its blocks share the weight, rounded up, beside its threads' 128
   operations and 64 for the axis before its row. The blocks are counted here
   one by one: rows of 97 and 104 bytes of uint8 (104 into a destination one
   byte past 16, where every row's end crosses), and of 49 uint16 and float32
   elements. *)
let test_step_serial () =
  let run = Nx_metal.Kernels.copy_run_bytes in
  List.iter
    (fun (D.Any dt, e, head) ->
      let w = D.bits dt / 8 in
      let rows = 4096 in
      let row = e * w and bytes = rows * e * w in
      let b = Plan.make () in
      let ws = Plan.workspace b in
      let site ~residue l =
        Nx_metal.Moves.site_at ~slot:ws ~base:0 ~residue (D.Any dt) l
      in
      let src =
        L.v
          ~strides:(Iarray.of_list [ (2 * e) + 2; 2 ])
          (Iarray.of_list [ rows; e ])
      in
      let dst = L.contiguous (Iarray.of_list [ rows; e ]) in
      ignore
        (Nx_metal.Moves.launch b ~n:(rows * e) ~dst:(site ~residue:head dst)
           (site ~residue:0 src));
      let p = Plan.finish b in
      let blocks = (head + bytes + run - 1) / run in
      let serial = ref 0 in
      for k = 0 to blocks - 1 do
        let lo = (k * run) - head in
        let hi = lo + run in
        let crossed = lo / row <> (hi - 1) / row in
        if lo < 0 || hi > bytes || (lo >= 0 && crossed) then incr serial
      done;
      let serial = !serial in
      let elements =
        (serial + Int.min serial ((blocks + 31) / 32)) * (run / w)
      in
      let share = ((elements * 512) + blocks - 1) / blocks in
      let threads =
        Array.fold_left
          (fun t (l : Plan.launch) ->
            let x, y, z = l.groups and per, _, _ = l.threads in
            t + (x * y * z * per))
          0 p.launches
      in
      let msg = Printf.sprintf "%s rows of %d bytes" (D.name dt) row in
      Array.iter
        (fun (l : Plan.launch) ->
          equal ~msg string
            (Printf.sprintf "copy_step_%d" (8 * w))
            (Nx_metal.Kernels.name l.kernel))
        p.launches;
      equal ~msg int
        ((threads * (128 + 64)) + (blocks * share))
        (Array.fold_left
           (fun t (l : Plan.launch) -> t + l.cost.ops)
           0 p.launches))
    [
      (D.Any D.Uint8, 97, 0);
      (D.Any D.Uint8, 104, 1);
      (D.Any D.Uint16, 49, 0);
      (D.Any D.Float32, 49, 4);
    ]

(* Tickets *)

let read_word b at =
  let s = Bytes.create 4 in
  Rig.Buffer.blit_to_bytes b at s 0 4;
  Bytes.get_int32_le s 0

let zeroed n =
  let b = Rig.Buffer.create (S.device ()) n in
  Rig.Buffer.fill b '\000';
  b

(* harness.metal's combine: groups of 256 threads, each with 257 words of
   threadgroup memory. *)
let combine_threads = 256
let combine_shared = 4 * (combine_threads + 1)

type tickets = { groups : int; rounds : int }

let pp_tickets ppf c =
  Format.fprintf ppf "%d groups, %d rounds" c.groups c.rounds

let tickets =
  let open Gen in
  with_pp pp_tickets
    (let+ groups =
       frequency
         [
           (2, constant 1);
           (2, constant 2);
           (6, int_range 3 32);
           (4, int_range 33 127);
           (2, constant 128);
         ]
     and+ rounds = int_range 1 6 in
     { groups; rounds })

(* One group, two and 128, each an eighth of the draws, so that each run reaches
   them. *)
let tickets_examples =
  [
    { groups = 1; rounds = 3 };
    { groups = 2; rounds = 1 };
    { groups = 128; rounds = 6 };
  ]

(* Of [groups] groups that each write their part and take one ticket, the last
   to take it sees every part of its round, one group is last each round, and
   the ticket is zero after each: device.h's protocol. A part read from an
   earlier round differs from its round's. *)
let law_tickets c =
  cover "one group" (c.groups = 1);
  cover "two groups" (c.groups = 2);
  cover "128 groups" (c.groups = 128);
  cover "several rounds" (c.rounds > 1);
  let parts = zeroed (4 * combine_threads * c.groups)
  and ticket = zeroed 4
  and out = zeroed 8 in
  for round = 1 to c.rounds do
    S.launch "combine" ~groups:c.groups ~threads:combine_threads
      ~shared:combine_shared
      [ S.A out; S.A parts; S.A ticket; S.W c.groups; S.W round ]
  done;
  equal ~msg:"parts read stale" int32 0l (read_word out 0);
  equal ~msg:"last groups" int32 (Int32.of_int c.rounds) (read_word out 4);
  equal ~msg:"ticket" int32 0l (read_word ticket 0)

(* Loops *)

(* The float32 values about zero and the least normal number, a few normal ones
   and the specials, as bits: every pair of them drawn by the two operands'
   arrays. 2^-63 squared and 2^-125 over -2 are 2^-126 exactly; 2^-126 by 1 -
   2^-24 lies below it and rounds to it. *)
let near_zero32 =
  [|
    0x0000_0000;
    0x8000_0000;
    0x0000_0001;
    0x8000_0001;
    0x0000_0002;
    0x003f_ffff;
    0x0040_0000;
    0x807f_ffff;
    0x007f_ffff;
    0x0080_0000;
    0x8080_0000;
    0x0080_0001;
    0x0100_0000;
    0x3380_0000;
    0x1f80_0000;
    0x2000_0000;
    0x3f7f_ffff;
    0x3f80_0000;
    0xc000_0000;
    0x3f00_0000;
    0x7f7f_ffff;
    0x7f80_0000;
    0xff80_0000;
    0x7fc0_0000;
    0xffc0_0001;
    0x7f80_0001;
  |]

(* A C-contiguous array of [dt] and [shape] on the host from the bits [vs]. *)
let of_bits_array (type v s) (dt : (v, s) D.t) shape vs =
  let w = D.bits dt / 8 and n = Array.length vs in
  let s =
    String.init (n * w) (fun i ->
        Char.chr ((vs.(i / w) lsr (8 * (i mod w))) land 0xff))
  in
  let x = A.create Rig.host dt (Iarray.of_list shape) in
  Rig.Buffer.blit_from_string s 0 (A.buffer x) 0 (String.length s);
  x

(* [vs] as the first operand of every pair, and as the second, of [dt] by
   [of_bits]. *)
let all_pairs (type v s) (dt : (v, s) D.t) of_bits vs =
  let n = Array.length vs in
  let arr f = of_bits_array dt [ n * n ] (Array.init (n * n) f) in
  (arr (fun i -> of_bits vs.(i / n)), arr (fun i -> of_bits vs.(i mod n)))

let bf16_of_bits b = (b lsr 16) land 0xffff

(* nx_metal.mli's subnormal rule at float32 and bfloat16: each kind of one
   or two operands at every pair of values about zero, read as zeros of
   their signs and a result below 2^-126 a zero of its sign, but the kinds
   whose result is an integer value, a boolean or an operand's bits, which
   are exact. *)
(* Rows of a dtype's name and what a law takes of it, each drawn, printed
   by its name. *)
let named rows =
  Gen.of_list ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name) rows

let flush_dtypes =
  [
    ("float32", (D.Any D.Float32, Fun.id));
    ("bfloat16", (D.Any D.Bfloat16, bf16_of_bits));
  ]

(* Every triple of [vs], as [all_pairs] gives pairs. *)
let all_triples (type v s) (dt : (v, s) D.t) of_bits vs =
  let n = Array.length vs in
  let arr f = of_bits_array dt [ n * n * n ] (Array.init (n * n * n) f) in
  ( arr (fun i -> of_bits vs.(i / (n * n))),
    arr (fun i -> of_bits vs.(i / n mod n)),
    arr (fun i -> of_bits vs.(i mod n)) )

(* Fma at every triple of [x], [y] and [z]: its operands read flushed, its exact
   value rounded once, its product unflushed, its result flushed. Covers the
   paths of bfloat16's fma on Metal (nx_kinds.h): a zero factor, an addend past
   2^60 of the product, a product past 2^60 of the addend, a result below
   2^-126, a result past the greatest finite value. *)
let law_fma_flush (b : Support.backend) x y z =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let values a =
    Array.map
      (fun v -> if Float.abs v < 0x1p-126 then 0. else v)
      (A.to_array (Kernels_cases.reference a D.Float64))
  in
  let xs = values x and ys = values y and zs = values z in
  let finite v = Float.is_finite v in
  Array.iteri
    (fun i a ->
      let c = zs.(i) and p = a *. ys.(i) in
      let exact = finite a && finite ys.(i) && finite c in
      cover "a zero factor" (exact && p = 0.);
      cover "an addend past 2^60 of the product"
        (exact && p <> 0. && Float.abs p < Float.abs c *. 0x1p-60);
      cover "a product past 2^60 of the addend"
        (exact && c <> 0. && Float.abs c < Float.abs p *. 0x1p-60);
      cover "a result below 2^-126"
        (exact && p +. c <> 0. && Float.abs (p +. c) < 0x1p-126);
      cover "a result past the greatest finite value"
        (exact && Float.abs (p +. c) > 0x1.fffffep127))
    xs;
  Kernels_cases.answers b (K3 Fma) (D.Any dt) (D.Any dt) shape
    ~accepted:(P.accepts3 Fma dt dt)
    ~want:(fun () ->
      Kernels_cases.expected_on b "fma" dt ~compare:false [| x; y; z |])
    (fun dst ->
      Support.map3
        (module K)
        Fma ~dst:(A.expect dt (A.Any dst)) (Support.on b x) (Support.on b y)
        (Support.on b z))

let test_flush (_, (D.Any dt, of_bits)) =
  let b = backend () in
  let x, y = all_pairs dt of_bits near_zero32 in
  (* At float32: a product of 2^-126, and one below it that rounds to it, which
     the rule flushes. *)
  (match D.Any dt with
  | D.Any D.Float32 ->
      let xs = A.to_array (Kernels_cases.reference x D.Float64) in
      let ys = A.to_array (Kernels_cases.reference y D.Float64) in
      let products = Array.mapi (fun i a -> Float.abs (a *. ys.(i))) xs in
      let rounded p = Float.abs (Int32.float_of_bits (Int32.bits_of_float p)) in
      cover "a product of exactly 2^-126"
        (Array.exists (fun p -> p = 0x1p-126) products);
      cover "a product below 2^-126 that rounds to it"
        (Array.exists (fun p -> p < 0x1p-126 && rounded p = 0x1p-126) products)
  | D.Any _ -> ());
  Kernels_cases.law_map1 b (Kernels_cases.Case x);
  Kernels_cases.law_map2 b (Kernels_cases.Pair (x, y));
  let x, y, z = all_triples dt of_bits near_zero32 in
  law_fma_flush b x y z

(* Every bfloat16 NaN, both signs: 254 of them, and the values about zero. *)
let bf16_order_values =
  Array.append
    (Array.map bf16_of_bits near_zero32)
    (Array.init 254 (fun i ->
         let sign = if i >= 127 then 0x8000 else 0 in
         sign lor 0x7f80 lor (1 + (i mod 127))))

(* The kinds that order values compare their keys: at the neighbours of zero and
   every bfloat16 NaN, Maximum, Minimum, the comparisons, Sign, Floor and Ceil
   give nx.cpu's bits, and Max, Min, Arg Max and Arg Min reductions nx.cpu's
   results, any NaN for a NaN. *)
type ordered = Ordered : ('v, 's) D.t * int array -> ordered

let order_dtypes =
  [
    ("bfloat16", Ordered (D.Bfloat16, bf16_order_values));
    ("float32", Ordered (D.Float32, near_zero32));
  ]

let test_order (_, Ordered (dt, vs)) =
  let b = backend () in
  let kinds1 =
    List.filter
      (fun ((u : P.unary), _) ->
        match u with Sign | Floor | Ceil -> true | _ -> false)
      Kernels_cases.unaries
  in
  let kinds2 =
    List.filter
      (function
        | P.Binary (Maximum | Minimum) | Compare _ -> true | Binary _ -> false)
      Kernels_cases.op2s
  in
  let maps (type v s) (dt : (v, s) D.t) vs =
    let x, y = all_pairs dt Fun.id vs in
    Kernels_cases.law_map1 ~kinds:kinds1 b (Kernels_cases.Case x);
    Kernels_cases.law_map2 ~kinds:kinds2 b (Kernels_cases.Pair (x, y))
  in
  (* Reductions of rows, each a row of the values in another order. *)
  let module K = (val b.kernels) in
  let reduce (type v s) (dt : (v, s) D.t) vs =
    let n = Array.length vs in
    let rows =
      Array.init n (fun r -> Array.init n (fun i -> vs.((i + r) mod n)))
    in
    let x = of_bits_array dt [ n; n ] (Array.concat (Array.to_list rows)) in
    let shape = Iarray.of_list [ n; n ] in
    let id = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |] in
    List.iter
      (fun (r : Nx_kernel.reduction) ->
        let k = Nx_metal.Loop.destinations r in
        let dts =
          if k = 2 then [| D.Any dt; D.Any D.Int64 |] else [| D.Any dt |]
        in
        let call (module K : Nx_kernel.S) dev x =
          let ds =
            Array.map
              (fun (D.Any d) -> A.Any (A.create dev d (Iarray.of_list [ n ])))
              dts
          in
          equal answer A.Done
            (K.reduce id ~shape ~axes:[| 1 |]
               [| (r, 0) |]
               ~dsts:(Iarray.of_array ds) [| Nx_kernel.Plain x |]);
          ds
        in
        let want = call (module Nx_cpu) Rig.host x in
        (* By the direct kernel, and by the tape where bound. *)
        List.iter
          (fun direct ->
            let got, traced =
              Fun.protect
                ~finally:(fun () -> Nx_metal.bound_direct true)
                (fun () ->
                  Nx_metal.bound_direct direct;
                  Plan.traced (fun () ->
                      call (module K) b.device (Support.on b x)))
            in
            equal ~msg:"by the direct kernel where not bound" bool direct
              (reduce_directs traced <> []);
            same_bits want got)
          [ true; false ])
      [ Monoid Max; Monoid Min; Arg Max; Arg Min ]
  in
  maps dt vs;
  reduce dt vs

(* Tickets: a sum of [groups] groups of 16 blocks each, of [outputs] outputs, in
   [rounds] rounds of other terms, has nx.cpu's bits: each output's last group
   sees every group's state, and its ticket is zero for the next call. Under a
   budget below an output's groups, its groups run apart and a launch merges
   them. A sum of a load runs a direct kernel, loop_reduce_direct at int32 and
   loop_row_sum_f32 at float32, or the tape where [direct] is [false]. *)
type summed = {
  groups : int;
  outputs : int;
  rounds : int;
  apart : bool;
  direct : bool;
  int32 : bool;
}

let pp_summed ppf c =
  Format.fprintf ppf "%d groups, %d outputs, %d rounds%s%s%s" c.groups c.outputs
    c.rounds
    (if c.apart then ", apart" else "")
    (if c.direct then "" else ", by the tape")
    (if c.int32 then ", int32" else "")

let summed =
  let open Gen in
  with_pp pp_summed
    (let+ groups =
       frequency
         [
           (3, constant 1);
           (2, constant 2);
           (6, int_range 3 32);
           (4, int_range 33 127);
           (3, constant 128);
         ]
     and+ outputs = frequency [ (3, constant 1); (1, int_range 2 3) ]
     and+ rounds = int_range 1 3
     and+ apart = bool
     and+ direct = bool
     and+ int32 = bool in
     { groups; outputs; rounds; apart; direct; int32 })

let block_terms = 16 * 1024

(* The groups of a plan's reduction launches: more than its groups where a
   group's runs take several launches. *)
let reduce_groups (p : Plan.t) =
  Array.fold_left
    (fun n (l : Plan.launch) ->
      if is_reduce (Nx_metal.Kernels.name l.kernel) then
        let g, _, _ = l.groups in
        n + g
      else n)
    0 p.launches

(* The least budget, within 0.1 %, under which a sum of [c]'s shape is planned
   with its groups whole: one group of an output fits it, and an output's
   groups, of more work, run apart. *)
let least_budget ~holds c =
  let (D.Any dt) = if c.int32 then D.Any D.Int32 else D.Any D.Float32 in
  let x =
    broadcast dt
      (Iarray.of_list [ c.outputs; c.groups * block_terms ])
      (Rig.Buffer.create Rig.host 16)
  in
  let d =
    broadcast dt (Iarray.of_list [ c.outputs ]) (Rig.Buffer.create Rig.host 16)
  in
  let id = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |] in
  let fits t =
    bounded t (fun () ->
        match
          loop_reduce ~holds id
            ~shape:(Iarray.of_list [ c.outputs; c.groups * block_terms ])
            ~axes:[| 1 |]
            [| (Monoid Sum, 0) |]
            ~dsts:[| A.Any d |] [| Nx_kernel.Plain x |]
        with
        | Ok (Some p) -> reduce_groups p = c.outputs * c.groups
        | Ok None | Error _ -> false)
  in
  let rec go lo hi =
    if hi /. lo < 1.001 then hi
    else
      let m = sqrt (lo *. hi) in
      if fits m then go lo m else go m hi
  in
  go 1e-7 Plan.budget

let law_summed c =
  let b = backend () in
  let terms = c.groups * block_terms in
  cover "one group" (c.groups = 1);
  cover "128 groups" (c.groups = 128);
  cover "several outputs" (c.outputs > 1);
  cover "apart" (c.apart && c.groups > 1);
  cover "by the tape" (not c.direct);
  Fun.protect ~finally:(fun () -> Nx_metal.bound_direct true) @@ fun () ->
  Nx_metal.bound_direct c.direct;
  let st = Random.State.make [| c.groups; c.outputs; c.rounds |] in
  let shape = Iarray.of_list [ c.outputs; terms ] in
  let (D.Any dt) = if c.int32 then D.Any D.Int32 else D.Any D.Float32 in
  let id = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |] in
  let sum (module K : Nx_kernel.S) dev x =
    let d = A.create dev dt (Iarray.of_list [ c.outputs ]) in
    equal answer A.Done
      (K.reduce id ~shape ~axes:[| 1 |]
         [| (Monoid Sum, 0) |]
         ~dsts:[| A.Any d |] [| Nx_kernel.Plain x |]);
    d
  in
  for _ = 1 to c.rounds do
    let x = A.create Rig.host dt shape in
    let n = Rig.Buffer.length (A.buffer x) in
    let bytes = Bytes.create n in
    for i = 0 to (n / 4) - 1 do
      let v =
        if c.int32 then Random.State.bits32 st
        else Int32.bits_of_float (Random.State.float st 2. -. 1.)
      in
      Bytes.set_int32_le bytes (4 * i) v
    done;
    Rig.Buffer.blit_from_string (Bytes.to_string bytes) 0 (A.buffer x) 0 n;
    let want = sum (module Nx_cpu) Rig.host x in
    let gx = A.to_device b.device x in
    let run () = Plan.traced (fun () -> sum (module Nx_metal) b.device gx) in
    let got, traced =
      if c.apart && c.groups > 1 then
        bounded (least_budget ~holds:(Nx_metal.holds b.device) c) run
      else run ()
    in
    let merged =
      List.exists
        (fun (t : Plan.traced) -> t.kernel = Nx_metal.Kernels.Loop_merge)
        traced
    in
    let ticketed = List.exists (fun (t : Plan.traced) -> t.tickets) traced in
    let took = reduce_directs traced <> [] in
    let by k = List.exists (fun (t : Plan.traced) -> t.kernel = k) traced in
    let generic = by Nx_metal.Kernels.Loop_reduce_direct
    and row = by Nx_metal.Kernels.Loop_row_sum_f32 in
    cover "merged by a launch" merged;
    cover "merged by a ticket" ticketed;
    cover "merged by a launch, by loop_reduce_direct" (merged && generic);
    cover "merged by a ticket, by loop_reduce_direct" (ticketed && generic);
    cover "merged by a launch, by loop_row_sum_f32" (merged && row);
    cover "merged by a ticket, by loop_row_sum_f32" (ticketed && row);
    equal ~msg:"by the direct kernel where not bound" bool c.direct took;
    equal ~msg:"merged apart under the bound" bool
      (c.apart && c.groups > 1)
      merged;
    equal (array int)
      (Kernels_cases.bits_of want)
      (Kernels_cases.bits_of (Support.host got))
  done

(* A reduction of no term into some elements declines where the reduction of one
   term does, at float64, and gives the identity elsewhere (loop.mli). *)
let test_no_term () =
  let b = backend () in
  let reduce (type v s) (dt : (v, s) D.t) r =
    let id = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |] in
    let x = A.create b.device dt (Iarray.of_list [ 3; 0 ]) in
    let y _ = A.Any (A.create b.device dt (Iarray.of_list [ 3 ])) in
    Nx_metal.reduce id ~shape:[| 3; 0 |] ~axes:[| 1 |]
      [| (r, 0) |]
      ~dsts:(Iarray.init (Nx_metal.Loop.destinations r) y)
      [| Nx_kernel.Plain x |]
  in
  List.iter
    (fun r ->
      equal ~msg:"float64" answer A.Declined (reduce D.Float64 r);
      equal ~msg:"float32" answer A.Done (reduce D.Float32 r))
    [ Nx_kernel.Logsumexp; Moments; Monoid Sum; Monoid Prod ]

(* Scans by their length *)

(* nx_metal.mli's [n] for a scan of [p]'s output 0: the nodes past its loads
   that the scanned value depends on, and one for a carry where [carried]. *)
let depends ?(carried = false) (p : P.t) =
  let seen = Array.make (Iarray.length p.nodes) false in
  let rec mark i =
    if not seen.(i) then begin
      seen.(i) <- true;
      match Iarray.get p.nodes i with
      | In _ | Coord _ | Const _ -> ()
      | Op1 (_, _, a) -> mark a
      | Op2 (_, a, b) ->
          mark a;
          mark b
      | Op3 (_, a, b, c) ->
          mark a;
          mark b;
          mark c
    end
  in
  mark (Iarray.get p.outs 0);
  let computed i s = match Iarray.get p.nodes i with In _ -> false | _ -> s in
  let n = ref (if carried then 1 else 0) in
  Array.iteri (fun i s -> if computed i s then incr n) seen;
  !n

(* The elements of the long scans the tests plan, for a value that depends on
   [n] nodes: [2^26 / (1 + n)], an estimate of seconds, in parts past 10 s. *)
let kept_elements n = (1 lsl 26) / (1 + n)

(* The most terms a slice the plans below draw, and the bytes of the host buffer
   their arrays share: a load of strides 1 reaches the sum of its extents, of at
   most 8 bytes an element. Plans read descriptors alone: the buffer is never
   read or written, so its pages are never resident. *)
let long_terms = 1 lsl 26
let long_bytes = 8 * (long_terms + 1024)
let long_buffer = lazy (Rig.Buffer.create Rig.host long_bytes)

(* The plan of [p]'s scan by [r] along axis 0 of [shape], from a carry where
   [carried], over loads of strides 1, which keep every axis of more than one
   element apart in its walk, into destinations of [into] whose results step by
   one along the slice where [step], else broadcast; at [holds] threads a group,
   by default 256, the fewest a loop's kernel runs with, at which a core holds
   one group. Where [bare], the loads too broadcast, so that one element of
   [buffer] holds a call of any length. *)
let long_plan ?(holds = 256) ?(carried = false) ?(bare = false) buffer p
    (r : Nx_kernel.reduction) ~into ~shape ~step =
  let rank = Array.length shape in
  let shape = Iarray.of_array shape in
  let ones = Iarray.init rank (fun _ -> if bare then 0 else 1) in
  let dst (D.Any d) =
    let strides = Iarray.init rank (fun i -> if step && i = 0 then 1 else 0) in
    A.Any (A.v d (L.v ~strides shape) buffer)
  in
  let dsts =
    match r with
    | Arg _ -> [| dst into; dst (D.Any D.Int64) |]
    | Moments -> [| dst into; dst into |]
    | Logsumexp | Monoid _ -> [| dst into |]
  in
  let loads =
    Iarray.map
      (fun (D.Any d) ->
        Nx_kernel.Plain (A.v d (L.v ~strides:ones shape) buffer))
      p.P.ins
  in
  let carry =
    if not carried then None
    else
      let (D.Any c) = Nx_kernel.carrier into in
      Some (A.Any (A.v c (L.v ~strides:ones shape) buffer))
  in
  loop_scan ?carry
    ~holds:(fun _ -> holds)
    p ~axis:0 (r, 0) ~dsts:(Iarray.of_array dsts) loads

let pp_shape ppf s =
  Format.pp_print_string ppf
    (String.concat "x" (Array.to_list (Array.map string_of_int s)))

let identity d = P.v ~ins:[| d |] [| P.In 0 |] ~outs:[| 0 |]

(* [n] Exps over a load of [d]. *)
let exps d n =
  P.v ~ins:[| d |]
    (Iarray.init (n + 1) (fun i ->
         if i = 0 then P.In 0 else P.Op1 (Unary Exp, d, i - 1)))
    ~outs:[| n |]

(* [n] Exps over a load of [d], each followed by a Log: 2n costly nodes whose
   values stay near the load's. *)
let exp_logs d n =
  P.v ~ins:[| d |]
    (Iarray.init
       ((2 * n) + 1)
       (fun i ->
         if i = 0 then P.In 0
         else if i mod 2 = 1 then P.Op1 (Unary Exp, d, i - 1)
         else P.Op1 (Unary Log, d, i - 1)))
    ~outs:[| 2 * n |]

(* [n] Pows over two loads of [d], each reading the second. *)
let pows d n =
  P.v ~ins:[| d; d |]
    (Iarray.init (n + 2) (fun i ->
         if i < 2 then P.In i else P.Op2 (Binary Pow, i - 1, 1)))
    ~outs:[| n + 1 |]

(* [n] Exps over the first of [loads] loads of [d], then the sum of the last Exp
   and the other loads: more operands than a scan's launch takes, so the scan
   runs in parts, whose first maps the Exps. *)
let loaded_exps d ~loads n =
  P.v
    ~ins:(Iarray.init loads (fun _ -> d))
    (Iarray.init
       (loads + n + loads - 1)
       (fun i ->
         if i < loads then P.In i
         else if i < loads + n then
           P.Op1 (Unary Exp, d, if i = loads then 0 else i - 1)
         else P.Op2 (Binary Add, i - 1, i - loads - n + 1)))
    ~outs:[| loads + n + loads - 2 |]

(* A load of [d] raised to itself. *)
let self_pow d =
  P.v ~ins:[| d |] [| P.In 0; P.Op2 (Binary Pow, 0, 0) |] ~outs:[| 1 |]

let fma d =
  P.v ~ins:[| d; d; d |]
    [| P.In 0; P.In 1; P.In 2; P.Op3 (Fma, 0, 1, 2) |]
    ~outs:[| 3 |]

type long_case = {
  what : string;
  p : P.t;
  r : Nx_kernel.reduction;
  into : D.any;
  carried : bool;
}

(* A scan of each class nx_metal.mli states a floor for, at the costliest kinds
   and dtypes of each: an Arg of a load narrower than a byte and not, a Monoid
   into a destination narrower than a byte and not, from a carry, programs of 1
   to 64 nodes of the costliest kinds over 1 to 3 loads, and Logsumexp and
   Moments of an operand and of programs; and each of a sum, an Arg, a Logsumexp
   and Moments at the most nodes it states, 2300 of a polynomial kind or 1800 of
   a complex one. A program of 64 costly nodes takes one group of a chunk's
   totals past a launch's budget, and one of 160 a block of 16 tiles. One of 10
   loads and 1000 Exps runs in parts, whose map of the Exps takes past a launch
   at 4 elements a thread. *)
let long_cases =
  let f32 = D.Any D.Float32 and bf16 = D.Any D.Bfloat16 in
  let i4 = D.Any D.Int4 and c64 = D.Any D.Complex64 in
  let case ?(carried = false) what p r into = { what; p; r; into; carried } in
  Nx_kernel.
    [
      case "an Arg of int4" (identity i4) (Arg Max) i4;
      case "an Arg of float32" (identity f32) (Arg Max) f32;
      case "an Arg of complex64" (identity c64) (Arg Max) c64;
      case "an int4 sum into int4" (identity i4) (Monoid Sum) i4;
      case "a float32 sum into int4" (identity f32) (Monoid Sum) i4;
      case "a float32 sum" (identity f32) (Monoid Sum) f32;
      case "a complex64 product" (identity c64) (Monoid Prod) c64;
      case ~carried:true "a float32 sum from a carry" (identity f32)
        (Monoid Sum) f32;
      case ~carried:true "an Arg of int4 from a carry" (identity i4) (Arg Max)
        i4;
      case "an Arg of a complex64 Exp" (exps c64 1) (Arg Max) c64;
      case "a sum of 2 complex64 Exps" (exps c64 2) (Monoid Sum) c64;
      case "an Arg of 8 complex64 Pows" (pows c64 8) (Arg Max) c64;
      case "a sum of 64 float32 Exps" (exps f32 64) (Monoid Sum) f32;
      case "an Arg of 64 complex64 Pows" (pows c64 64) (Arg Max) c64;
      case "a sum of 2300 float32 Exps" (exps f32 2300) (Monoid Sum) f32;
      case "an Arg of 1800 complex64 Pows" (pows c64 1800) (Arg Max) c64;
      case "a sum of 10 float32 loads and 1000 Exps"
        (loaded_exps f32 ~loads:10 1000)
        (Monoid Sum) f32;
      case "a sum of a bfloat16 Fma" (fma bf16) (Monoid Sum) bf16;
      case "a float32 Logsumexp" (identity f32) Logsumexp f32;
      case "a Logsumexp of 160 float32 Exps" (exps f32 160) Logsumexp f32;
      case "a Logsumexp of 2300 float32 Exps" (exps f32 2300) Logsumexp f32;
      case "float32 Moments" (identity f32) Moments f32;
      case "bfloat16 Moments" (identity bf16) Moments bf16;
      case "Moments of a bfloat16 Fma" (fma bf16) Moments bf16;
      case "Moments of 4 float32 Pows" (pows f32 4) Moments f32;
      case "Moments of 2300 float32 Exps" (exps f32 2300) Moments f32;
    ]

(* The number of the class [what] of [cases]. *)
let class_in cases what =
  let rec go i = function
    | [] -> invalid_arg what
    | c :: cs -> if c.what = what then i else go (i + 1) cs
  in
  go 0 cases

let long_class = class_in long_cases

(* The shape of [rank] axes whose first has [terms], its last [last] and the
   others 2. *)
let long_shape rank ~terms ~last =
  Array.init rank (fun i ->
      if i = 0 then terms else if i = rank - 1 then last else 2)

(* The terms a slice of a scan of [c] of [rank] axes, its others of 2, that
   [kept_elements] allow; 0 where they allow none. *)
let kept_terms c rank =
  kept_elements (depends ~carried:c.carried c.p) lsr (rank - 1)

let fail_declined c ~step ~holds shape a =
  failf "%s%s, %a, %d threads a group: %a" c.what
    (if step then ", stepping by one" else "")
    pp_shape shape holds Nx_array_support.pp_answer a

(* nx_metal.mli: a scan declines only where one element's program passes a
   launch. Each class at every rank to 32, its other axes of 2, at the most
   terms a slice [kept_elements] allow, its results stepping by one along the
   slice and not; past 26 axes of 2 no length is within them. *)
let test_kept_scans () =
  let buffer = Lazy.force long_buffer in
  List.iter
    (fun c ->
      List.iter
        (fun step ->
          for rank = 1 to 32 do
            let terms = kept_terms c rank in
            if terms > 0 then
              let shape = long_shape rank ~terms ~last:2 in
              match
                long_plan ~carried:c.carried buffer c.p c.r ~into:c.into ~shape
                  ~step
              with
              | Ok _ -> ()
              | Error a -> fail_declined c ~step ~holds:256 shape a
          done)
        [ false; true ])
    long_cases

(* nx_metal.mli: a loop's thread takes 1 element where a run of 4 would take a
   launch past 50 ms, so a map is declined only where one element's program
   does. Maps of 600, 1000 and 2300 float32 Exps, whose run of 4 passes a launch
   from about 600, over 2^16 elements. *)
let test_costly_maps () =
  let f32 = D.Any D.Float32 in
  let n = 1 lsl 16 in
  let array () =
    A.Any
      (A.v D.Float32
         (L.v ~strides:[| 1 |] [| n |])
         (Rig.Buffer.create Rig.host (4 * n)))
  in
  List.iter
    (fun nodes ->
      let (A.Any x) = array () in
      match
        loop_map
          ~holds:(fun _ -> 256)
          (exps f32 nodes)
          ~dsts:[| array () |]
          [| Nx_kernel.Plain x |]
      with
      | Ok (Some _) -> ()
      | Ok None -> failf "a map of %d Exps: no plan" nodes
      | Error a ->
          failf "a map of %d Exps: %a" nodes Nx_array_support.pp_answer a)
    [ 600; 1000; 2300 ]

(* nx_metal.mli: a scan declines only where one element's program passes a
   launch. Each of a float32 Logsumexp and float32 Moments of 2^28 elements, at
   every rank to 29, its other axes of 2, the 1-D slice of 2^28 terms first. *)
let test_network_lengths () =
  let buffer = Lazy.force long_buffer in
  let f32 = D.Any D.Float32 in
  List.iter
    (fun (what, r) ->
      for rank = 1 to 29 do
        let shape = long_shape rank ~terms:(1 lsl (29 - rank)) ~last:2 in
        match
          long_plan ~bare:true buffer (identity f32) r ~into:f32 ~shape
            ~step:false
        with
        | Ok _ -> ()
        | Error a ->
            failf "%s, %a: %a" what pp_shape shape Nx_array_support.pp_answer a
      done)
    Nx_kernel.
      [ ("a float32 Logsumexp", Logsumexp); ("float32 Moments", Moments) ]

(* nx_metal.mli's figure, from the estimate: a 1-D Logsumexp or Moments slice of
   2^24 terms takes 0.43 s where Metal lets a group have 256 threads, the fewest
   a loop's kernel runs with. Baselines of the estimate, of the networks also at
   576 threads, the network's on an M1 Max; of the sums at one slice, and at
   slices of one chunk and a term, the costliest shape of those tried. *)
let test_estimates () =
  let buffer = Lazy.force long_buffer in
  let f32 = D.Any D.Float32 in
  let seconds (p : Plan.t) =
    Array.fold_left
      (fun t (l : Plan.launch) ->
        t
        +. Plan.seconds
             ~threads:(Nx_metal.Kernels.threads l.kernel)
             ~groups:l.groups ~resident:l.resident l.cost)
      0. p.launches
  in
  let estimate ?(holds = 256) ?(into = f32) p r shape =
    match long_plan ~holds ~bare:true buffer p r ~into ~shape ~step:false with
    | Ok (Some p) -> Printf.sprintf "%.2f s" (seconds p)
    | Ok None -> "no plan"
    | Error _ -> "declined"
  in
  let network (what, r) holds =
    let at bits = estimate ~holds (identity f32) r [| 1 lsl bits |] in
    Printf.sprintf "%s, %d threads: 2^24 %s, 2^28 %s, 2^29 %s" what holds
      (at 24) (at 28) (at 29)
  in
  let networks =
    List.concat_map
      (fun c -> List.map (network c) [ 256; 576 ])
      Nx_kernel.[ ("Logsumexp", Logsumexp); ("Moments", Moments) ]
  in
  let sum ?(d = f32) n shape =
    let (D.Any dd) = d in
    Printf.sprintf "a sum of %d %s Exps, %s: %s" n (D.name dd)
      (Format.asprintf "%a" pp_shape shape)
      (estimate ~into:d (exps d n) (Monoid Sum) shape)
  in
  (* Slices of 4,097 terms are the costliest: a chunk of one term costs as much
     as a whole one. *)
  let sums =
    [
      sum 0 [| 4097; kept_elements 0 / 4097 |];
      sum 1 [| kept_elements 1 |];
      sum 4 [| 4097; kept_elements 4 / 4097 |];
      sum ~d:(D.Any D.Complex64) 4 [| 4097; kept_elements 4 / 4097 |];
      sum 2300 [| kept_elements 2300 |];
    ]
  in
  expect (String.concat "\n" (networks @ sums))
  @@ __POS_OF__
       {|
    Logsumexp, 256 threads: 2^24 0.43 s, 2^28 6.83 s, 2^29 13.67 s
    Logsumexp, 576 threads: 2^24 0.24 s, 2^28 3.84 s, 2^29 7.69 s
    Moments, 256 threads: 2^24 0.42 s, 2^28 6.65 s, 2^29 13.31 s
    Moments, 576 threads: 2^24 0.22 s, 2^28 3.52 s, 2^29 7.04 s
    a sum of 0 float32 Exps, 4097x16380: 1.60 s
    a sum of 1 float32 Exps, 33554432: 1.89 s
    a sum of 4 float32 Exps, 4097x3276: 5.27 s
    a sum of 4 complex64 Exps, 4097x3276: 6.01 s
    a sum of 2300 float32 Exps, 29165: 4.73 s
    |}

(* nx_metal.mli: a reduction's group is charged the lines its outputs' terms
   touch together. A float32 sum of [4; 2^24] along axis 0 takes groups of 64
   outputs side by side, whose 256 bytes a term lie in at most 3 lines; one of
   [2^24; 4] along axis 1 groups of 64 rows of 4 terms, whose 1 KiB lies in at
   most 9. Each of its row kernel's two passes is charged the operand's bytes
   times at most 3/2 and 9/8; past them the outputs' bytes, a quarter of the
   operand's, and less than a tenth more for its launches' states. A call is
   only planned. *)
let test_box_estimates () =
  let buffer = Lazy.force long_buffer in
  let outputs = Rig.Buffer.create Rig.host (4 * (1 lsl 24)) in
  let charged shape axis =
    let x = A.v D.Float32 (L.contiguous (Iarray.of_array shape)) buffer in
    let kept = [| shape.(1 - axis) |] in
    let z = A.v D.Float32 (L.contiguous (Iarray.of_array kept)) outputs in
    match
      loop_reduce
        ~holds:(fun _ -> 1024)
        (identity (D.Any D.Float32))
        ~shape:(Iarray.of_array shape) ~axes:(Iarray.of_list [ axis ])
        [| (Monoid Sum, 0) |]
        ~dsts:[| A.Any z |] [| Nx_kernel.Plain x |]
    with
    | Ok (Some p) ->
        let bytes =
          Array.fold_left
            (fun t (l : Plan.launch) -> t + l.cost.bytes)
            0 p.launches
        in
        Float.of_int bytes /. Float.of_int (4 * (1 lsl 26))
    | Ok None | Error _ -> Float.nan
  in
  less float_exact
    ~than:((2. *. 1.5) +. 0.25 +. 0.1)
    (charged [| 4; 1 lsl 24 |] 0);
  less float_exact
    ~than:((2. *. 1.125) +. 0.25 +. 0.1)
    (charged [| 1 lsl 24; 4 |] 1)

(* nx_metal.mli, Time: a direct reduction along a row is weighed by its kernel's
   work on a term, its operand's bytes counted twice, a [float16] operand's
   once, a Logsumexp's or Moments' in one launch with its first pass four times,
   an exp at 32 operations. The bench's rows of these shapes take the kernel of
   their reduction and dtype, a Sum, Max or Arg in one launch where a core holds
   1024 threads, a Logsumexp or Moments in 2 by their estimates. The estimates'
   milliseconds are baselines. *)
let test_row_estimates () =
  let buffer = Lazy.force long_buffer in
  let outputs = Rig.Buffer.create Rig.host (8 * 16384) in
  let estimate (D.Any d as dt) r shape =
    let rank = Array.length shape in
    let x = A.v d (L.contiguous (Iarray.of_array shape)) buffer in
    let rows = Iarray.of_array (Array.sub shape 0 (rank - 1)) in
    let dst (D.Any e) = A.Any (A.v e (L.contiguous rows) outputs) in
    let dsts =
      match r with
      | Nx_kernel.Arg _ -> [| dst dt; dst (D.Any D.Int64) |]
      | Moments -> [| dst dt; dst dt |]
      | _ -> [| dst (Nx_kernel.carrier dt) |]
    in
    match
      loop_reduce
        ~holds:(fun _ -> 1024)
        (identity dt) ~shape:(Iarray.of_array shape)
        ~axes:(Iarray.of_list [ rank - 1 ])
        [| (r, 0) |]
        ~dsts:(Iarray.of_array dsts) [| Nx_kernel.Plain x |]
    with
    | Ok (Some p) ->
        let s =
          Array.fold_left
            (fun t (l : Plan.launch) ->
              t
              +. Plan.seconds
                   ~threads:(Nx_metal.Kernels.threads l.kernel)
                   ~groups:l.groups ~resident:l.resident l.cost)
            0. p.launches
        in
        let bytes =
          Array.fold_left
            (fun t (l : Plan.launch) -> t + l.cost.bytes)
            0 p.launches
        in
        let n = Array.length p.launches in
        let operand = numel shape * D.bytes d 1 in
        let counted = bytes / operand in
        let line =
          Printf.sprintf "%s of %s %s: %.1f ms"
            (Nx_metal.Kernels.name p.launches.(0).kernel)
            (D.name d)
            (Format.asprintf "%a" pp_shape shape)
            (s *. 1e3)
        in
        (* The outputs' bytes and the states' are a few thousandths of the
           operand's, and a group's rows, which may start anywhere in a line,
           one line more than theirs: 2 rows of 8,192 bfloat16s are charged 257
           lines, 0.4% past 256, each time they count. Past the counted times by
           at most a fiftieth. *)
        less float_exact
          ~than:(Float.of_int counted +. 0.02)
          (Float.of_int bytes /. Float.of_int operand);
        ((Nx_metal.Kernels.name p.launches.(0).kernel, n, counted), line)
    | Ok None -> (("no plan", 0, 0), "")
    | Error _ -> (("declined", 0, 0), "")
  in
  let rows =
    Nx_kernel.
      [
        estimate (D.Any D.Float32) (Arg Max) [| 8192; 8192 |];
        estimate (D.Any D.Float32) (Monoid Sum) [| 8192; 8192 |];
        estimate (D.Any D.Bfloat16) (Monoid Sum) [| 16384; 8192 |];
        estimate (D.Any D.Float16) (Monoid Max) [| 8192; 8192 |];
        estimate (D.Any D.Float32) Logsumexp [| 8192; 8192 |];
        estimate (D.Any D.Float32) Moments [| 8192; 8192 |];
      ]
  in
  (* Each row's kernel, its launches and the times its operand's bytes count. *)
  equal
    (list (triple string int int))
    [
      ("loop_row_argmax_f32", 1, 2);
      ("loop_row_sum_f32", 1, 2);
      ("loop_row_sum_bf16", 1, 2);
      ("loop_row_max_f16", 1, 1);
      ("loop_row_logsumexp_f32", 2, 4);
      ("loop_row_moments_f32", 2, 4);
    ]
    (List.map fst rows);
  expect (String.concat "\n" (List.map snd rows))
  @@ __POS_OF__
       {|
    loop_row_argmax_f32 of float32 8192x8192: 28.1 ms
    loop_row_sum_f32 of float32 8192x8192: 28.1 ms
    loop_row_sum_bf16 of bfloat16 16384x8192: 49.0 ms
    loop_row_max_f16 of float16 8192x8192: 14.9 ms
    loop_row_logsumexp_f32 of float32 8192x8192: 72.1 ms
    loop_row_moments_f32 of float32 8192x8192: 63.3 ms
    |}

(* The lines of Plan.line bytes that [a]'s elements touch, from a buffer that
   starts on a line. *)
let touched (type a b) (a : (a, b) A.t) =
  let l = A.layout a and w = D.bytes (A.dtype a) 1 in
  let shape = Iarray.to_array (L.shape l) in
  let strides = Iarray.to_array (L.strides l) in
  let seen = Hashtbl.create 4096 in
  let rec go k pos =
    if k = Array.length shape then begin
      Hashtbl.replace seen (pos * w / Plan.line) ();
      Hashtbl.replace seen ((((pos + 1) * w) - 1) / Plan.line) ()
    end
    else
      for i = 0 to shape.(k) - 1 do
        go (k + 1) (pos + (i * strides.(k)))
      done
  in
  go 0 (L.offset l);
  Hashtbl.length seen

let test_short_runs () =
  let f32 = D.Float32 and n = 4096 and long = 1 lsl 20 in
  let buffer = Rig.Buffer.create Rig.host (8 * long) in
  let view ~offset ~strides shape =
    A.v f32
      (L.v ~offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape))
      buffer
  in
  let charged = function
    | Ok (Some (p : Plan.t)) ->
        Array.fold_left
          (fun t (l : Plan.launch) -> t + l.cost.bytes)
          0 p.launches
    | Ok None | Error _ -> 0
  in
  let holds _ = 1024 in
  let add =
    P.v ~ins:[| D.Any f32; D.Any f32 |]
      [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
      ~outs:[| 2 |]
  in
  (* An add of two views of [shape] into a contiguous destination past them; the
     lines its operands touch and the bytes its launches count. *)
  let map ~offset ~strides shape =
    let x = view ~offset ~strides shape in
    let z = view ~offset:long ~strides:[| shape.(1); 1 |] shape in
    let lines = (2 * touched x) + touched z in
    let p = loop_map ~holds add ~dsts:[| A.Any z |] [| Plain x; Plain x |] in
    (lines * Plan.line, charged p)
  in
  (* A sum of a view of [shape] along its last axis. *)
  let sum ~offset ~strides shape =
    let x = view ~offset ~strides shape in
    let z = view ~offset:long ~strides:[| 1 |] [| shape.(0) |] in
    let p =
      loop_reduce ~holds (identity (D.Any f32)) ~shape:(Iarray.of_array shape)
        ~axes:[| 1 |]
        [| (Nx_kernel.Monoid Sum, 0) |]
        ~dsts:[| A.Any z |] [| Plain x |]
    in
    (touched x * Plan.line, charged p)
  in
  let rows = [| 64; 1 |] and pair = [| n; 2 |] in
  (* Two floats a row of 64: at bytes 124 to 131 of each row, two lines a row,
     and at 0 to 7, one. *)
  List.iter
    (fun (lines, bytes) -> at_least int ~than:lines bytes)
    [
      map ~offset:31 ~strides:rows pair;
      map ~offset:0 ~strides:rows pair;
      sum ~offset:31 ~strides:rows pair;
      sum ~offset:0 ~strides:rows pair;
    ];
  (* Windows of 32 floats a float apart: their lines, a line past each operand's
     span at most, though every window's run touches 2. *)
  let lines, bytes = map ~offset:0 ~strides:[| 1; 1 |] [| n; 32 |] in
  at_least int ~than:lines bytes;
  at_most int ~than:(lines + (4 * Plan.line)) bytes;
  (* One contiguous run: its lines, past them by a line's share alone. *)
  let lines, bytes = map ~offset:0 ~strides:[| long; 1 |] [| 1; long |] in
  at_least int ~than:lines bytes;
  at_most int ~than:(lines + (lines / 100)) bytes

(* The bytes past which a line touched again may have left the cache: the least
   system level cache of an Apple7 GPU, the M1's 8 MB. AnandTech measured the M1
   Pro's at 24 MB and the Max's at 48 MB, against the M1's 8 MB ("Apple's M1
   Pro, M1 Max SoCs Investigated", 2021); the Ultra is two Max dies ("Apple
   Announces M1 Ultra", 2022), so 96 MB. The cases below touch lines 64 MiB
   apart: past every cache but the Ultra's, which holds them, so there they move
   fewer lines than they are charged. *)
let cache = 8 lsl 20

(* The bytes the launches of a map of x + x count, x of [shape] at strides [xs]
   from the start of [buffer], into a contiguous destination from element
   [offset]. *)
let far_map buffer ~xs ~shape ~offset =
  let f32 = D.Float32 and shape = Iarray.of_array shape in
  let twice =
    P.v ~ins:[| D.Any f32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |]
  in
  let x = A.v f32 (L.v ~strides:(Iarray.of_array xs) shape) buffer in
  let z =
    A.v f32 (L.v ~offset ~strides:(L.strides (L.contiguous shape)) shape) buffer
  in
  match
    loop_map ~holds:(fun _ -> 1024) twice ~dsts:[| A.Any z |] [| Plain x |]
  with
  | Ok (Some (p : Plan.t)) ->
      Array.fold_left (fun t (l : Plan.launch) -> t + l.cost.bytes) 0 p.launches
  | Ok None -> fail "no plan"
  | Error _ -> fail "declined"

(* x, of 2^19 rows of 32 floats, a line a row, its first 31 of each read
   transposed into 31 rows of 2^19, too few to stage: each of x's lines is
   touched once a row of the destination, after 2^19 of x's other lines, 64 MiB,
   so each of the 31 2^19 elements read moves a line, but on an Ultra. *)
let test_far_transpose () =
  let rows = 1 lsl 19 in
  let n = 32 * rows in
  assert (rows * Plan.line > cache);
  let buffer = Rig.Buffer.create Rig.host (8 * n) in
  at_least int
    ~than:(31 * rows * Plan.line)
    (far_map buffer ~xs:[| 1; 32 |] ~shape:[| 31; rows |] ~offset:n)

(* nx_metal.mli: a tile's staged load is charged its lines along b. All 32 of
   x's columns read transposed, staged: each of x's lines is read whole by one
   group, and x and the destination are each charged at most twice their bytes,
   a run's lines and the next. *)
let test_staged_transpose () =
  let rows = 1 lsl 19 in
  let n = 32 * rows in
  let buffer = Rig.Buffer.create Rig.host (8 * n) in
  at_most int
    ~than:(2 * (2 * n * 4))
    (far_map buffer ~xs:[| 1; 32 |] ~shape:[| 32; rows |] ~offset:n)

(* y, of 2^24 floats, 64 MiB, broadcast along 3 rows: each of its lines is
   touched once a row, after all of y's others, so it moves y's lines 3 times,
   but on an Ultra, and the destination its own. *)
let test_far_broadcast () =
  let m = 3 and cols = 1 lsl 24 in
  assert (cols * 4 > cache);
  let buffer = Rig.Buffer.create Rig.host (4 * (cols + (m * cols))) in
  at_least int
    ~than:(2 * m * cols * 4)
    (far_map buffer ~xs:[| 0; 1 |] ~shape:[| m; cols |] ~offset:cols)

(* Staged operands

   nx_metal.mli: a map stages a plain load of 8, 16 or 32 bits whose stride is 1
   on the walk's axis before the last of extent past 1 and at least 32 elements
   along the last, both extents at least 32, through threadgroup memory, a
   group's 32 x 32 tile at a time; its bits are the unstaged map's. The cases
   map x + y, or x mod y where asked, y cast to x's dtype where they differ,
   plus a broadcast row or the last coordinate where asked, into a contiguous
   [batch; rows; cols] destination of x's dtype, or into x itself; or, for a
   staged operand's raw bits, x + x and y as it is. Each operand is contiguous
   or the transpose of a contiguous [batch; cols; pitch] array, its first [rows]
   of each [pitch], y's columns reversed or at an odd offset where asked. *)
type staged = {
  st_x : D.any;
  st_y : D.any;
  st_batch : int;
  st_rows : int;
  st_cols : int;
  st_pitch : int;
  st_tx : bool;
  st_ty : bool;
  st_neg : bool;
  st_skip : int;
  st_extra : [ `None | `Bcast | `Coord ];
  st_raw : bool;
  st_alias : bool;
  st_mod : bool;
  st_seed : int;
}

let pp_staged ppf c =
  let side t = if t then "t" else "c" in
  Format.fprintf ppf "%s(%a) %s %s(%a)%s%s [%d; %d; %d] pitch %d%s%s%s seed %d"
    (side c.st_tx) Kernels_cases.pp_dtype c.st_x
    (if c.st_mod then "mod" else "+")
    (side c.st_ty) Kernels_cases.pp_dtype c.st_y
    (if c.st_neg then " reversed" else "")
    (if c.st_skip > 0 then " +1" else "")
    c.st_batch c.st_rows c.st_cols c.st_pitch
    (match c.st_extra with
    | `None -> ""
    | `Bcast -> " + row"
    | `Coord -> " + coord")
    (if c.st_raw then ", y out" else "")
    (if c.st_alias then ", into x" else "")
    c.st_seed

let staged_dtypes =
  D.[ Any Float32; Any Float16; Any Bfloat16; Any Uint8; Any Int32; Any Int64 ]

let staged_cases =
  let open Gen in
  with_pp pp_staged
  @@ let* st_x = of_list staged_dtypes in
     let* same = frequency [ (2, constant true); (1, constant false) ] in
     let* st_y = if same then constant st_x else of_list staged_dtypes in
     let* st_batch = of_list [ 1; 1; 3 ] in
     let* st_rows = of_list [ 1; 2; 31; 32; 33; 63; 64; 100 ] in
     let* st_cols = of_list [ 1; 31; 32; 33; 63; 64; 100 ] in
     let* past = of_list [ 0; 0; 0; 9; 40 ] in
     let* st_tx = bool in
     let* st_ty = frequency [ (2, constant true); (1, constant false) ] in
     let* st_neg = frequency [ (3, constant false); (1, constant true) ] in
     let* st_skip = of_list [ 0; 0; 1 ] in
     let* st_extra =
       frequency
         [ (3, constant `None); (1, constant `Bcast); (1, constant `Coord) ]
     in
     let* st_raw = frequency [ (4, constant false); (1, constant true) ] in
     let* st_alias = frequency [ (4, constant false); (1, constant true) ] in
     let* st_mod = frequency [ (3, constant false); (1, constant true) ] in
     let+ st_seed = int_range 0 1_000_000 in
     {
       st_x;
       st_y;
       st_batch;
       st_rows;
       st_cols;
       st_pitch = st_rows + past;
       st_tx = st_tx && not st_alias;
       st_ty;
       st_neg = st_neg && st_ty;
       st_skip = (if st_ty then st_skip else 0);
       st_extra;
       st_raw;
       st_alias;
       st_mod;
       st_seed;
     }

let staged_prog c =
  let ins =
    match c.st_extra with
    | `Bcast -> [| c.st_x; c.st_y; c.st_x |]
    | `None | `Coord -> [| c.st_x; c.st_y |]
  in
  let nodes = ref [ P.In 1; P.In 0 ] in
  let add n =
    nodes := n :: !nodes;
    List.length !nodes - 1
  in
  let k = if c.st_mod then P.Mod else Add in
  let sum =
    if c.st_raw then add (Op2 (Binary Add, 0, 0))
    else if c.st_x = c.st_y then add (Op2 (Binary k, 0, 1))
    else
      let y = add (Op1 (Cast, c.st_x, 1)) in
      add (Op2 (Binary k, 0, y))
  in
  let last =
    match c.st_extra with
    | `None -> sum
    | `Bcast ->
        let w = add (In 2) in
        add (Op2 (Binary Add, sum, w))
    | `Coord ->
        let k = add (Coord 2) in
        let k = add (Op1 (Cast, c.st_x, k)) in
        add (Op2 (Binary Add, sum, k))
  in
  let outs = if c.st_raw then [| last; 1 |] else [| last |] in
  P.v ~ins:(Iarray.of_array ins)
    (Iarray.of_list (List.rev !nodes))
    ~outs:(Iarray.of_array outs)

(* A host array of [dt] and [shape] drawn from [st]: floats in [-1, 1) unless
   [raw], else bytes as they come, NaN payloads and subnormals among them. *)
let staged_fill st (D.Any d) shape ~raw =
  let n = numel shape in
  let floats () = Array.init n (fun _ -> Random.State.float st 2. -. 1.) in
  let shape = Iarray.of_array shape in
  let bytes () =
    let x = A.create Rig.host d shape in
    let b = A.buffer x in
    let len = Rig.Buffer.length b in
    Rig.Buffer.blit_from_string (random_bytes st len) 0 b 0 len;
    A.Any x
  in
  if raw then bytes ()
  else
    match d with
    | D.Float32 -> A.Any (A.of_array d shape (floats ()))
    | D.Float16 -> A.Any (A.of_array d shape (floats ()))
    | D.Bfloat16 -> A.Any (A.of_array d shape (floats ()))
    | _ -> bytes ()

(* Operand [y] or [x] of [c] on [dev], drawn from [st]. *)
let staged_operand c st dev ~y =
  let t = if y then c.st_ty else c.st_tx in
  let (D.Any d as dt) = if y then c.st_y else c.st_x in
  let b = c.st_batch and r = c.st_rows and k = c.st_cols in
  let q = c.st_pitch in
  let skip = if y then c.st_skip else 0 in
  let base = if t then [| b; k; q |] else [| b; r; k |] in
  let len = numel base + skip in
  let (A.Any h) = staged_fill st dt [| len |] ~raw:(y && c.st_raw) in
  let h = A.expect d (A.Any h) in
  let on = if Rig.equal dev Rig.host then h else A.to_device dev h in
  let shape = Iarray.of_list [ b; r; k ] in
  let layout =
    if not t then
      L.v ~offset:skip ~strides:(L.strides (L.contiguous shape)) shape
    else if y && c.st_neg then
      L.v
        ~offset:(skip + ((k - 1) * q))
        ~strides:(Iarray.of_list [ k * q; 1; -q ])
        shape
    else L.v ~offset:skip ~strides:(Iarray.of_list [ k * q; 1; q ]) shape
  in
  A.Any (A.v d layout (A.buffer on))

(* [c]'s destinations and loads on [dev]. *)
let staged_call dev c =
  let st = Random.State.make [| c.st_seed |] in
  let x = staged_operand c st dev ~y:false in
  let y = staged_operand c st dev ~y:true in
  let (D.Any e) = c.st_x in
  let shape = Iarray.of_list [ c.st_batch; c.st_rows; c.st_cols ] in
  let z = if c.st_alias then x else A.Any (A.create dev e shape) in
  let dsts =
    if c.st_raw then
      let (D.Any f) = c.st_y in
      [| z; A.Any (A.create dev f shape) |]
    else [| z |]
  in
  let (A.Any xa) = x in
  let (A.Any ya) = y in
  let loads =
    match c.st_extra with
    | `Bcast ->
        let (A.Any w) = staged_fill st c.st_x [| c.st_cols |] ~raw:false in
        let w = if Rig.equal dev Rig.host then w else A.to_device dev w in
        [|
          Nx_kernel.Plain xa;
          Plain ya;
          Plain
            (A.v (A.dtype w)
               (L.v ~strides:(Iarray.of_list [ 0; 0; 1 ]) shape)
               (A.buffer w));
        |]
    | `None | `Coord -> [| Nx_kernel.Plain xa; Plain ya |]
  in
  (Iarray.of_array dsts, Iarray.of_array loads)

let staged_run (module K : Nx_kernel.S) dev c =
  let dsts, loads = staged_call dev c in
  let a = K.map (staged_prog c) ~dsts loads in
  equal ~msg:(Format.asprintf "%a" pp_staged c) answer A.Done a;
  Iarray.to_array dsts

(* The most threadgroup memory a launch of [c]'s host plan takes. *)
let case_shared c =
  let dsts, loads = staged_call Rig.host c in
  match loop_map ~holds:(fun _ -> 1024) (staged_prog c) ~dsts loads with
  | Ok (Some (p : Plan.t)) ->
      Array.fold_left
        (fun m (l : Plan.launch) -> Int.max m l.shared)
        0 p.launches
  | Ok None -> fail "no plan"
  | Error _ -> fail "declined"

(* nx_metal.mli's Memory: a staged load's stage, past the group's planes. *)
let stage = 4 * Nx_metal.Kernels.loop_stage_words

(* The planes of a map's group whose plan with no stage takes [plain] bytes: a
   direct map's group holds none, and 16 bytes where it stages nothing. *)
let planes_of plain = if plain = 16 then 0 else plain

(* The loads [c]'s host plan stages, as nx_metal.mli's Memory counts them: a
   stage each past the planes of the same map of contiguous operands. *)
let staged_loads c =
  let plain =
    { c with st_tx = false; st_ty = false; st_neg = false; st_skip = 0 }
  in
  (case_shared c - planes_of (case_shared plain)) / stage

(* The dtypes of [c]'s loads its plan stages, where its count tells them: none,
   or every transposed one. *)
let staged_dtypes_of c k =
  let ts =
    List.filter_map
      (fun (dt, t) -> if t then Some dt else None)
      [ (c.st_x, c.st_tx); (c.st_y, c.st_ty) ]
  in
  if k = List.length ts then ts else []

let staged_case ?(x = D.Any D.Float32) ?y ?(batch = 1) ?(tx = false)
    ?(ty = true) ?(neg = false) ?(skip = 0) ?(extra = `None) ?(raw = false)
    ?(alias = false) ?(md = false) ?pitch rows cols =
  {
    st_x = x;
    st_y = Option.value y ~default:x;
    st_batch = batch;
    st_rows = rows;
    st_cols = cols;
    st_pitch = Option.value pitch ~default:rows;
    st_tx = tx;
    st_ty = ty;
    st_neg = neg;
    st_skip = skip;
    st_extra = extra;
    st_raw = raw;
    st_alias = alias;
    st_mod = md;
    st_seed = rows + cols;
  }

let staged_examples =
  let c = staged_case in
  let f16 = D.Any D.Float16 and u8 = D.Any D.Uint8 and i64 = D.Any D.Int64 in
  let bf16 = D.Any D.Bfloat16 in
  [
    (* x + t(y) and t(x) + y, whole tiles, 32 bits, direct; x mod t(y), of the
       long kinds' kernel. *)
    c 64 64;
    c ~tx:true ~ty:false 64 64;
    c ~md:true 64 64;
    (* Two staged loads. *)
    c ~tx:true 64 64;
    (* 16 and 8 bits. *)
    c ~x:f16 64 64;
    c ~x:u8 64 64;
    (* Each narrow float's tiled kernels, of the cheap kinds and the long
       ones. *)
    c ~x:bf16 64 64;
    c ~x:bf16 ~md:true 64 64;
    c ~x:f16 ~md:true 64 64;
    (* The threshold, |s_a| = 32, and one under it. *)
    c 32 64;
    c 31 64;
    (* Extents 32, 33 and 63 along a; 33 and 63 along b. *)
    c 33 33;
    c 64 32;
    c 64 33;
    c 64 63;
    c 33 64;
    c 63 64;
    (* Rows of 40 and of 16 at a stride of 64 along a. *)
    c ~pitch:64 40 64;
    c ~pitch:64 16 64;
    (* Stepping back along a; at an odd offset, its quads unaligned. *)
    c ~neg:true 64 64;
    c ~skip:1 64 64;
    (* Batched. *)
    c ~batch:3 64 33;
    (* Beside a broadcast row and a contiguous operand; a coordinate. *)
    c ~extra:`Bcast 64 64;
    c ~extra:`Coord 64 64;
    (* Through the tape: a float16 operand cast into float32. *)
    c ~y:f16 64 64;
    (* The staged operand's raw bits out. *)
    c ~raw:true 64 64;
    (* Into x itself. *)
    c ~alias:true 64 64;
    (* A width the rule excludes. *)
    c ~x:i64 64 64;
  ]

(* nx_metal.mli: a map whose loads are staged has nx.cpu's bits. Its covers
   count the loads its host plan stages. *)
let law_staged c =
  let b = backend () in
  let k = staged_loads c in
  let staged = staged_dtypes_of c k in
  let transposed = c.st_tx || c.st_ty in
  let bits w = List.exists (fun (D.Any d) -> D.bits d = w) staged in
  cover "a staged load of 8 bits" (bits 8);
  cover "a staged load of 16 bits" (bits 16);
  cover "a staged load of 32 bits" (bits 32);
  List.iter
    (fun e ->
      cover
        (Printf.sprintf "a staged load, %d along a" e)
        (k >= 1 && c.st_cols = e);
      cover
        (Printf.sprintf "a staged load, %d along b" e)
        (k >= 1 && c.st_rows = e))
    [ 32; 33; 63 ];
  cover "a staged load at a stride of 32 along a" (k >= 1 && c.st_pitch = 32);
  cover "a staged slice of a transpose" (k >= 1 && c.st_pitch > c.st_rows);
  cover "a transposed load one under the threshold, unstaged"
    (transposed && k = 0 && c.st_pitch = 31 && c.st_cols >= 32);
  cover "a slice of a transpose under 32 rows, unstaged"
    (transposed && k = 0 && c.st_rows < 32 && c.st_pitch >= 32
   && c.st_cols >= 32);
  cover "a staged load stepping back" (k >= 1 && c.st_ty && c.st_neg);
  cover "a staged load at an odd offset" (k >= 1 && c.st_ty && c.st_skip > 0);
  cover "two staged loads" (k = 2);
  cover "a staged load beside a broadcast and a contiguous operand"
    (k >= 1 && c.st_extra = `Bcast && not c.st_tx);
  cover "a transposed load beside a coordinate, at 1 element a thread, unstaged"
    (transposed && k = 0 && c.st_extra = `Coord && c.st_rows >= 32
   && c.st_cols >= 32);
  cover "a staged load in a batch" (k >= 1 && c.st_batch > 1);
  cover "a staged load in a direct map"
    (k >= 1 && c.st_x = c.st_y && c.st_extra = `None && not c.st_raw);
  cover "a staged load in a tape map"
    (k >= 1 && (c.st_x <> c.st_y || c.st_extra <> `None || c.st_raw));
  cover "a staged operand's bits out" (k >= 1 && c.st_ty && c.st_raw);
  cover "a staged load into a destination identical to a load"
    (k >= 1 && c.st_alias);
  cover "a transposed load of 64 bits, unstaged"
    (c.st_ty && k = 0
    && (let (D.Any d) = c.st_y in
        D.bits d)
       = 64
    && c.st_rows >= 32 && c.st_cols >= 32);
  let direct = c.st_x = c.st_y && c.st_extra = `None && not c.st_raw in
  let words32 =
    direct
    &&
    let (D.Any d) = c.st_x in
    D.bits d = 32
  in
  let words16 =
    if direct && (c.st_x = D.Any D.Bfloat16 || c.st_x = D.Any D.Float16) then
      Some c.st_x
    else None
  in
  let words8 =
    direct
    &&
    let (D.Any d) = c.st_x in
    D.bits d = 8
  in
  let want = staged_run (module Nx_cpu) Rig.host c in
  let got, traced =
    Plan.traced (fun () -> staged_run (module Nx_metal) b.device c)
  in
  check_direct32 ?words16 ~words8 ~words32 ~long:c.st_mod traced;
  cover "the 32-bit tiled kernel" (ran Loop_map_direct32_tiled traced);
  cover "the tiled kernel of bytes" (ran Loop_map_direct8_tiled traced);
  cover "the bfloat16 tiled kernel" (ran Loop_map_direct16_bf16_tiled traced);
  cover "the bfloat16 tiled kernel of long kinds"
    (ran Loop_map_direct16_bf16_long_tiled traced);
  cover "the float16 tiled kernel" (ran Loop_map_direct16_f16_tiled traced);
  cover "the float16 tiled kernel of long kinds"
    (ran Loop_map_direct16_f16_long_tiled traced);
  cover "the 32-bit tiled kernel of long kinds"
    (ran Loop_map_direct32_long_tiled traced);
  cover "a 32-bit tiled direct map without quads"
    (words32 && ran Loop_map_direct_tiled traced);
  same_bits want got

(* The most threadgroup memory a launch of the host plan of x + y takes, x
   contiguous of [shape], y of [dt] at [strides] from [offset], read through a
   padding of nothing where [pad]. *)
let staged_shared ?(pad = false) (D.Any d as dt) shape ~strides ~offset =
  let shape = Iarray.of_array shape in
  let rank = Iarray.length shape in
  let reach =
    Array.fold_left ( + ) 0
      (Array.mapi (fun k s -> (Iarray.get shape k - 1) * abs s) strides)
  in
  let x = A.create Rig.host d shape and z = A.create Rig.host d shape in
  let bytes = (((offset + reach + 1) * D.bits d) + 7) / 8 in
  let y =
    A.v d
      (L.v ~offset ~strides:(Iarray.of_array strides) shape)
      (Rig.Buffer.create Rig.host bytes)
  in
  let zeros = Iarray.init rank (fun _ -> 0) in
  let ly =
    if pad then
      Nx_kernel.Padded
        {
          x = y;
          fill = D.zero d;
          pad = { lo = zeros; hi = zeros; interior = zeros; windows = [||] };
        }
    else Plain y
  in
  let p =
    P.v ~ins:[| dt; dt |]
      [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
      ~outs:[| 2 |]
  in
  match
    loop_map ~holds:(fun _ -> 1024) p ~dsts:[| A.Any z |] [| Plain x; ly |]
  with
  | Ok (Some (p : Plan.t)) ->
      Array.fold_left
        (fun m (l : Plan.launch) -> Int.max m l.shared)
        0 p.launches
  | Ok None -> fail "no plan"
  | Error _ -> fail "declined"

(* nx_metal.mli: a staged load takes a stage of threadgroup memory past the
   planes of the same map of contiguous operands; a map whose loads the rule
   does not stage takes what that map takes. On host plans of x + y at [rows;
   cols], y's stride 1 along the rows. *)
let test_staged_plans () =
  let f32 = D.Any D.Float32 in
  let check ?(dt = f32) ?pad what ~staged rows cols strides offset =
    let shape = [| rows; cols |] in
    let plain = staged_shared ?pad dt shape ~strides:[| cols; 1 |] ~offset:0 in
    let shared = staged_shared ?pad dt shape ~strides ~offset in
    let want = if staged then planes_of plain + stage else plain in
    equal ~msg:what int want shared
  in
  check "a transpose of 32 bits" ~staged:true 64 64 [| 1; 64 |] 0;
  check ~dt:(D.Any D.Float16) "a transpose of 16 bits" ~staged:true 64 64
    [| 1; 64 |] 0;
  check ~dt:(D.Any D.Uint8) "a transpose of bytes" ~staged:true 64 64
    [| 1; 64 |] 0;
  check "a transpose at the threshold" ~staged:true 32 64 [| 1; 32 |] 0;
  check "a transpose stepping back" ~staged:true 64 64 [| 1; -64 |] (63 * 64);
  check "a transpose in edge tiles" ~staged:true 33 63 [| 1; 33 |] 0;
  check "40 rows of a transpose at a stride of 64" ~staged:true 40 64
    [| 1; 64 |] 0;
  check "16 rows of a transpose at a stride of 64" ~staged:false 16 64
    [| 1; 64 |] 0;
  check "31 rows of a transpose at a stride of 64" ~staged:false 31 64
    [| 1; 64 |] 0;
  check "a stride of 31 along the columns" ~staged:false 64 64 [| 1; 31 |] 0;
  check "a transpose under the threshold" ~staged:false 31 64 [| 1; 31 |] 0;
  check "a transpose of 16 columns" ~staged:false 64 16 [| 1; 64 |] 0;
  check "a broadcast along the run" ~staged:false 64 64 [| 1; 0 |] 0;
  check "a step of 2 along the rows" ~staged:false 64 64 [| 2; 128 |] 0;
  check ~dt:(D.Any D.Int64) "a transpose of 64 bits" ~staged:false 64 64
    [| 1; 64 |] 0;
  check ~dt:(D.Any D.Complex64) "a transpose of complex64" ~staged:false 64 64
    [| 1; 64 |] 0;
  check ~dt:(D.Any D.Int4) "a transpose of int4" ~staged:false 64 64 [| 1; 64 |]
    0;
  check ~pad:true "a padded transpose" ~staged:false 64 64 [| 1; 64 |] 0

(* The sum of [n] float32 loads of [rows; cols] on [dev], load [k] transposed
   where [t k], then [exps] Exps: its program, destination and loads, drawn from
   [seed]. *)
let sum_call ?(exps = 0) ?(seed = 7) dev ~t n rows cols =
  let f32 = D.Float32 in
  let st = Random.State.make [| seed |] in
  let shape = Iarray.of_list [ rows; cols ] in
  let load k =
    let h =
      A.of_array f32
        (Iarray.of_list [ rows * cols ])
        (Array.init (rows * cols) (fun _ -> Random.State.float st 1.))
    in
    let on = if Rig.equal dev Rig.host then h else A.to_device dev h in
    let strides = if t k then [ 1; rows ] else [ cols; 1 ] in
    Nx_kernel.Plain
      (A.v f32 (L.v ~strides:(Iarray.of_list strides) shape) (A.buffer on))
  in
  let loads = Array.init n load in
  let add k = P.Op2 (Binary Add, (if k = 0 then 0 else n + k - 1), k + 1) in
  let top = (2 * n) - 2 in
  let exp k = P.Op1 (Unary Exp, D.Any f32, top + k) in
  let p =
    P.v
      ~ins:(Iarray.init n (fun _ -> D.Any f32))
      (Iarray.of_list
         (List.init n (fun k -> P.In k)
         @ List.init (n - 1) add
         @ List.init exps exp))
      ~outs:(Iarray.of_list [ top + exps ])
  in
  (p, A.Any (A.create dev f32 shape), Iarray.of_array loads)

(* The launches of the host plan of [sum_call]. *)
let sum_launches ?exps ~t n rows cols =
  let p, z, loads = sum_call ?exps Rig.host ~t n rows cols in
  match loop_map ~holds:(fun _ -> 1024) p ~dsts:[| z |] loads with
  | Ok (Some (p : Plan.t)) -> p.launches
  | Ok None -> fail "no plan"
  | Error _ -> fail "declined"

let most_shared ls =
  Array.fold_left (fun m (l : Plan.launch) -> Int.max m l.shared) 0 ls

(* nx_metal.mli: a map of 4 elements a thread takes tiles where its group's
   values and stages fit in 32 KiB and its estimate with them is below its
   estimate without. On sums of loads of 256 x 256. *)
let test_tile_choice () =
  let none _ = false and all _ = true and second k = k = 1 in
  let launches ?exps t n = sum_launches ?exps ~t n 256 256 in
  let shared ?exps t n = most_shared (launches ?exps t n) in
  let kernel ?exps t n =
    Nx_metal.Kernels.name (launches ?exps t n).(0).kernel
  in
  (* Three loads, each held in a plane of 4 KiB, and their stages. *)
  equal ~msg:"three transposed loads" int
    (planes_of (shared none 3) + (3 * stage))
    (shared all 3);
  (* A fourth load's plane takes the group's values past 16 KiB: 1 element a
     thread. *)
  equal ~msg:"four loads' kernel" string "loop_map_v1" (kernel all 4);
  equal ~msg:"four transposed loads" int (shared none 4) (shared all 4);
  (* A transpose's stage beside 3 planes of 4 KiB cuts the groups a core holds
     from 4 to 3: under 8 Exps a group's chain costs more in the third more
     waves than the stage saves in bytes. Under none, the stage saves more. *)
  equal ~msg:"x + t(y) under Exps, its kernel" string "loop_map_v4"
    (kernel ~exps:8 second 2);
  equal ~msg:"x + t(y) under Exps" int (shared ~exps:8 none 2)
    (shared ~exps:8 second 2);
  equal ~msg:"x + t(y) + z" int
    (planes_of (shared none 3) + stage)
    (shared second 3)

(* nx_metal.mli: a program too wide for a launch runs in parts, a part that
   reads a staged load taking tiles; the parts give nx.cpu's bits. *)
let test_staged_part () =
  let b = backend () in
  let n = 12 and rows = 64 and cols = 96 in
  let t k = k = 5 in
  let ls = sum_launches ~t n rows cols in
  let tiled = Array.exists (fun (l : Plan.launch) -> l.shared >= stage) ls in
  equal ~msg:"a part of tiles" bool true tiled;
  let run (module K : Nx_kernel.S) dev =
    let p, z, loads = sum_call dev ~t n rows cols in
    equal answer A.Done (K.map p ~dsts:[| z |] loads);
    [| z |]
  in
  same_bits (run (module Nx_cpu) Rig.host) (run (module Nx_metal) b.device)

(* nx_metal.mli: a program too wide for a launch runs in parts, and the parts
   give nx.cpu's bits. The powers x, x^2, ..., x^n of one load are all made
   before the first is summed, so [n] values live at once, past the planes a
   group holds and past the 64 a tape starts with. *)
let test_wide_values () =
  let b = backend () in
  let n = 70 and len = 1024 in
  let f32 = D.Float32 in
  let call dev =
    let st = Random.State.make [| 11 |] in
    let h =
      A.of_array f32 (Iarray.of_list [ len ])
        (Array.init len (fun _ -> 0.9 +. Random.State.float st 0.1))
    in
    let x = if Rig.equal dev Rig.host then h else A.to_device dev h in
    let power k = if k = 1 then P.In 0 else P.Op2 (Binary Mul, k - 2, 0) in
    let sum k = P.Op2 (Binary Add, (if k = 0 then 0 else n + k - 1), k + 1) in
    let p =
      P.v ~ins:[| D.Any f32 |]
        (Iarray.of_list
           (List.init n (fun k -> power (k + 1)) @ List.init (n - 1) sum))
        ~outs:[| (2 * n) - 2 |]
    in
    ( p,
      A.Any (A.create dev f32 (Iarray.of_list [ len ])),
      Iarray.of_list [ Nx_kernel.Plain x ] )
  in
  let p, z, loads = call Rig.host in
  (match loop_map ~holds:(fun _ -> 1024) p ~dsts:[| z |] loads with
  | Ok (Some (c : Plan.t)) ->
      at_least ~msg:"launches" int ~than:2 (Array.length c.launches)
  | Ok None -> fail "no plan"
  | Error _ -> fail "declined");
  let run (module K : Nx_kernel.S) dev =
    let p, z, loads = call dev in
    equal answer A.Done (K.map p ~dsts:[| z |] loads);
    [| z |]
  in
  same_bits (run (module Nx_cpu) Rig.host) (run (module Nx_metal) b.device)

(* Padded maps

   nx_metal.mli: a map's bits are nx.cpu's. The cases sum [n] padded loads of
   one dtype, plus a contiguous one where [plain], into a contiguous destination
   of [shape]: each load's own padding of its own source, read reversed along
   its last axis where asked, through a window on the last axis where there is
   one load. A load may pad by 2^31 and more before its first axis and crop as
   much after, so that its padded coordinates pass 2^31 while every read lands
   in fill. *)
type padded = {
  pd_dt : D.any;
  pd_shape : int array;
  pd_pads : Nx_kernel.pad array;
  pd_srcs : int array array;
  pd_rev : bool array;
  pd_plain : bool;
  pd_fill : float;
  pd_seed : int;
}

let pp_padded ppf c =
  let ints a = String.concat "x" (List.map string_of_int (Array.to_list a)) in
  let pad (p : Nx_kernel.pad) =
    Printf.sprintf "lo %s hi %s in %s%s"
      (ints (Iarray.to_array p.lo))
      (ints (Iarray.to_array p.hi))
      (ints (Iarray.to_array p.interior))
      (String.concat ""
         (List.map
            (fun (w : M.window) ->
              Printf.sprintf " w%d:%dx%d+%d" w.axis w.size w.step w.dilation)
            (Iarray.to_list p.windows)))
  in
  Format.fprintf ppf "%a [%s] of %d loads%s fill %g seed %d: %s"
    Kernels_cases.pp_dtype c.pd_dt (ints c.pd_shape) (Array.length c.pd_pads)
    (if c.pd_plain then " + plain" else "")
    c.pd_fill c.pd_seed
    (String.concat "; "
       (Array.to_list
          (Array.mapi
             (fun k p ->
               Printf.sprintf "%s%s from %s" (pad p)
                 (if c.pd_rev.(k) then " reversed" else "")
                 (ints c.pd_srcs.(k)))
             c.pd_pads)))

let padded_dtypes =
  D.[ Any Uint8; Any Int8; Any Float16; Any Float32; Any Int32; Any Int64 ]

(* A source extent and a padding of it to [d]: lo, hi, interior. Two thirds of
   the other axes are the source itself, so that most quads' other coordinates
   reach an element. Along the [last] axis half are, so that a quad reads 4
   elements whole; of the rest, two thirds have an interior, and most cross a
   pad's end within a quad. *)
let pad_axis ~last d =
  let open Gen in
  let* same =
    frequency [ (1, constant false); ((if last then 1 else 2), constant true) ]
  in
  if same then constant (d, 0, 0, 0)
  else
    let* interior =
      frequency [ (3, constant 0); ((if last then 6 else 1), int_range 1 2) ]
    in
    let* n = int_range 1 (d + 2) in
    let+ lo = int_range (-2) 3 in
    let hi = d - lo - (n + ((n - 1) * interior)) in
    (n, lo, hi, interior)

let far = (1 lsl 31) + 7

let rec all_of = function
  | [] -> Gen.constant []
  | g :: gs ->
      let open Gen in
      let* x = g in
      let+ xs = all_of gs in
      x :: xs

let padded_cases =
  let open Gen in
  with_pp pp_padded
  @@
  (* int64 half as often: its loads take 1 element a thread. *)
  let* pd_dt =
    frequency
      (List.map
         (fun dt -> ((if dt = D.Any D.Int64 then 1 else 2), constant dt))
         padded_dtypes)
  in
  let* rank = int_range 1 5 in
  (* Rows of a multiple of 4 most often, which take quads. *)
  let* last =
    frequency [ (1, of_list [ 1; 2; 3; 5; 7; 9 ]); (5, of_list [ 4; 8; 12 ]) ]
  in
  let* outer = array ~size:(constant (rank - 1)) (int_range 1 4) in
  let pd_shape = Array.append outer [| last |] in
  (* Two loads of no plain one take quads up to rank 4: 3 planes and the
     coordinates fit 16 KiB. *)
  let* n =
    frequency
      [ (8, constant 1); (5, constant 2); (1, constant 9); (2, constant 27) ]
  in
  let* windowed = frequency [ (8, constant false); (1, constant true) ] in
  let n = if windowed then 1 else n in
  let* pd_plain =
    if windowed || n = 2 then constant false
    else frequency [ (3, constant false); (1, constant true) ]
  in
  let* farthest = frequency [ (6, constant false); (1, constant true) ] in
  let one =
    map Array.of_list
      (all_of
         (Array.to_list
            (Array.mapi (fun j -> pad_axis ~last:(j = rank - 1)) pd_shape)))
  in
  let* axes = array ~size:(constant n) one in
  let* pd_rev =
    array ~size:(constant n)
      (frequency [ (5, constant false); (1, constant true) ])
  in
  let* size = int_range 1 3 in
  let* step = int_range 1 3 in
  let* dilation = int_range 1 2 in
  let* negative = bool in
  let pd_fill = if negative && pd_dt <> D.Any D.Uint8 then -3. else 3. in
  let+ pd_seed = int_range 0 1_000_000 in
  let pd_srcs = Array.map (Array.map (fun (n, _, _, _) -> n)) axes in
  let pd_pads =
    Array.mapi
      (fun k ax ->
        let lo = Array.map (fun (_, lo, _, _) -> lo) ax in
        let hi = Array.map (fun (_, _, hi, _) -> hi) ax in
        if farthest && k = 0 then begin
          lo.(0) <- lo.(0) + far;
          hi.(0) <- hi.(0) - far
        end;
        {
          Nx_kernel.lo = Iarray.of_array lo;
          hi = Iarray.of_array hi;
          interior = Iarray.of_array (Array.map (fun (_, _, _, i) -> i) ax);
          windows =
            (if windowed then
               Iarray.of_list [ { M.axis = rank - 1; size; step; dilation } ]
             else Iarray.of_list []);
        })
      axes
  in
  { pd_dt; pd_shape; pd_pads; pd_srcs; pd_rev; pd_plain; pd_fill; pd_seed }

(* A load of int64, which takes 1 element a thread, and one of int8 at a
   negative fill, 9 % and 8 % of the draws, so that each run reaches them: 3 x 8
   elements shifted by one along the last axis. *)
let padded_examples =
  let c pd_dt pd_fill pd_seed =
    let ints l = Iarray.of_list l in
    {
      pd_dt;
      pd_shape = [| 3; 8 |];
      pd_pads =
        [|
          {
            Nx_kernel.lo = ints [ 0; 1 ];
            hi = ints [ 0; -1 ];
            interior = ints [ 0; 0 ];
            windows = ints [];
          };
        |];
      pd_srcs = [| [| 3; 8 |] |];
      pd_rev = [| false |];
      pd_plain = false;
      pd_fill;
      pd_seed;
    }
  in
  [ c (D.Any D.Int64) 3. 1; c (D.Any D.Int8) (-3.) 2 ]

(* The destination's shape: the first load's padded shape. *)
let padded_shape c =
  match Nx_kernel.Shape.padded c.pd_pads.(0) c.pd_srcs.(0) with
  | Ok s -> s
  | Error e -> failf "padded_cases drew %s" e

let padded_prog c =
  let n = Array.length c.pd_pads in
  let k = if c.pd_plain then n + 1 else n in
  let nodes =
    List.init k (fun i -> P.In i)
    @ List.init (k - 1) (fun i ->
        P.Op2 (Binary Add, (if i = 0 then 0 else k - 1 + i), i + 1))
  in
  P.v
    ~ins:(Iarray.of_list (List.init k (fun _ -> c.pd_dt)))
    (Iarray.of_list nodes)
    ~outs:(Iarray.of_list [ List.length nodes - 1 ])

(* [c]'s destination and loads on [dev]. *)
let padded_call dev c =
  let st = Random.State.make [| c.pd_seed |] in
  let (D.Any d as dt) = c.pd_dt in
  let on (A.Any h) =
    let h = A.expect d (A.Any h) in
    if Rig.equal dev Rig.host then h else A.to_device dev h
  in
  let source k =
    let s = c.pd_srcs.(k) in
    let x = on (staged_fill st dt s ~raw:false) in
    if not c.pd_rev.(k) then x
    else
      let r = Array.length s in
      let strides =
        Iarray.to_array (L.strides (L.contiguous (Iarray.of_array s)))
      in
      strides.(r - 1) <- -1;
      A.v d
        (L.v
           ~offset:(s.(r - 1) - 1)
           ~strides:(Iarray.of_array strides) (Iarray.of_array s))
        (A.buffer x)
  in
  let fill = D.of_float d c.pd_fill in
  let loads =
    Array.mapi
      (fun k pad -> Nx_kernel.Padded { x = source k; fill; pad })
      c.pd_pads
  in
  let shape = padded_shape c in
  let loads =
    if c.pd_plain then
      let y = on (staged_fill st dt shape ~raw:false) in
      Array.append loads [| Nx_kernel.Plain y |]
    else loads
  in
  let z = A.create dev d (Iarray.of_array shape) in
  ([| A.Any z |], loads)

let padded_run (module K : Nx_kernel.S) dev c =
  let dsts, loads = padded_call dev c in
  equal
    ~msg:(Format.asprintf "%a" pp_padded c)
    answer A.Done
    (K.map (padded_prog c) ~dsts:(Iarray.of_array dsts) (Iarray.of_array loads));
  dsts

(* The launches of [c]'s host plan, none for an empty call. *)
let padded_launches c =
  let dsts, loads = padded_call Rig.host c in
  match
    loop_map
      ~holds:(fun _ -> 1024)
      (padded_prog c) ~dsts:(Iarray.of_array dsts) (Iarray.of_array loads)
  with
  | Ok (Some (p : Plan.t)) -> p.launches
  | Ok None -> [||]
  | Error _ -> fail "declined"

let quads_of (l : Plan.launch) =
  let mode = Bytes.get_int64_le l.words (8 * Nx_metal.Kernels.loop_mode) in
  Int64.logand mode (Int64.of_int Nx_metal.Kernels.loop_quads) <> 0L

(* Whether a quad of the walk's last axis, of extent [d], reads a padded axis of
   extent [n] by [lo] and [interior]: [`Whole] where some quad's 4 elements are
   elements, adjacent and on a multiple of 4 of a row whose length is too,
   [`Edge] where some quad has elements and fills on either side of the axis's
   ends. *)
let quad_reads ~d ~n ~lo ~interior =
  let s = interior + 1 in
  let inside j =
    let u = j - lo in
    u >= 0 && u mod s = 0 && u / s < n
  in
  let whole = ref false and edge = ref false in
  let i = ref 0 in
  while !i + 4 <= d do
    let ins = List.filter inside (List.init 4 (fun v -> !i + v)) in
    let k = List.length ins in
    if k = 4 && interior = 0 && (!i - lo) mod 4 = 0 && n mod 4 = 0 then
      whole := true;
    let beyond v = !i + v - lo < 0 || !i + v - lo > (n - 1) * s in
    if k > 0 && List.exists beyond [ 0; 1; 2; 3 ] then edge := true;
    i := !i + 4
  done;
  (!whole, !edge)

(* Whether a padded axis of extent [n] by [lo] and [interior] has an element at
   some coordinate of [0, d). *)
let reaches ~d ~n ~lo ~interior =
  let rec go m =
    m < n
    &&
    let c = lo + (m * (interior + 1)) in
    (c >= 0 && c < d) || go (m + 1)
  in
  go 0

let law_padded c =
  let b = backend () in
  let ls = padded_launches c in
  let any f = Array.exists f ls in
  let kernel k (l : Plan.launch) = l.kernel = k in
  let (D.Any d) = c.pd_dt in
  let some = Array.fold_left ( * ) 1 (padded_shape c) > 0 in
  let window = Iarray.length c.pd_pads.(0).windows > 0 in
  let interior =
    Array.exists
      (fun (p : Nx_kernel.pad) -> Iarray.exists (( < ) 0) p.interior)
      c.pd_pads
  in
  let far = Iarray.get c.pd_pads.(0).lo 0 >= far in
  let quads =
    some
    && any (fun l -> kernel Nx_metal.Kernels.Loop_map_padded l && quads_of l)
  in
  let rank = Array.length c.pd_shape in
  (* What a padded load's quads read along the last axis, where its other axes
     reach an element. *)
  let reads k =
    let p = c.pd_pads.(k) and src = c.pd_srcs.(k) in
    let axis j = (src.(j), Iarray.get p.lo j, Iarray.get p.interior j) in
    let others = ref true in
    for j = 0 to rank - 2 do
      let n, lo, interior = axis j in
      if not (reaches ~d:c.pd_shape.(j) ~n ~lo ~interior) then others := false
    done;
    let n, lo, interior = axis (rank - 1) in
    let whole, edge = quad_reads ~d:c.pd_shape.(rank - 1) ~n ~lo ~interior in
    (!others && whole && not c.pd_rev.(k), !others && edge)
  in
  let quad_reads =
    if (not quads) || window then [||]
    else Array.init (Array.length c.pd_pads) reads
  in
  cover "padded quads, a vector read" (Array.exists fst quad_reads);
  cover "padded quads, a quad across a pad's end" (Array.exists snd quad_reads);
  cover "two padded loads under quads" (quads && Array.length c.pd_pads >= 2);
  cover "padded quads, an interior along the last axis"
    (quads
    && Array.exists
         (fun (p : Nx_kernel.pad) -> Iarray.get p.interior (rank - 1) > 0)
         c.pd_pads);
  cover "int8 at 4 elements a thread, a negative fill"
    (some && c.pd_dt = D.Any D.Int8 && c.pd_fill < 0.
    && any (kernel Nx_metal.Kernels.Loop_map_padded));
  cover "padded quads"
    (some
    && any (fun l -> kernel Nx_metal.Kernels.Loop_map_padded l && quads_of l));
  cover "padded, 4 elements a thread, no quads"
    (some
    && any (fun l ->
        kernel Nx_metal.Kernels.Loop_map_padded l && not (quads_of l)));
  cover "64-bit padded loads, 1 element a thread"
    (some && D.bits d = 64 && any (kernel Nx_metal.Kernels.Loop_map_v1));
  cover "padded coordinates past 2^31, wide"
    (some && far && any (kernel Nx_metal.Kernels.Loop_map_wide));
  cover "27 padded loads" (some && Array.length c.pd_pads = 27);
  cover "a window" (some && window);
  cover "an interior" (some && interior);
  cover "a reversed source" (some && Array.exists Fun.id c.pd_rev);
  same_bits
    (padded_run (module Nx_cpu) Rig.host c)
    (padded_run (module Nx_metal) b.device c)

(* Maps along one axis

   A map whose walk is one row takes quads (tape.h) whatever its length: its
   quads end only at the row's end, where a thread has fewer than 4 elements. A
   map of one element, or of none of its axes past 1, walks one axis of extent
   1. A coordinate is an int64, so a map that reads one takes an element a
   thread, and no quads. *)

type axis_kind = Direct | Tape | Coordinate | Padding

type axis_map = {
  ak : axis_kind;
  adt : D.any;
  ashape : int array;
  askew : int;
  abcast : bool;
  alo : int;
  asrc : int;
  aseed : int;
}

let pp_axis_map ppf c =
  let kind =
    match c.ak with
    | Direct -> "a direct add"
    | Tape -> "an add on the tape"
    | Coordinate -> "an add of a coordinate"
    | Padding ->
        Printf.sprintf "an add of a padding (lo %d, from %d)" c.alo c.asrc
  in
  Format.fprintf ppf "%s at %a over %a, skew %d%s, seed %d" kind
    Kernels_cases.pp_dtype c.adt pp_shape c.ashape c.askew
    (if c.abcast then ", a broadcast load" else "")
    c.aseed

(* Mostly lengths of no multiple of 4, some past a group's 1024 elements; for
   the direct and tape kinds, also no axis, one axis of extent 1 before the row,
   and two of extent 1. *)
let axis_map =
  let open Gen in
  with_pp pp_axis_map
  @@
  let extent =
    frequency
      [
        (4, int_range 1 11);
        (2, of_list [ 1023; 1025; 4095; 4097; 4098 ]);
        (1, of_list [ 4; 8; 12 ]);
      ]
  in
  let* ak = of_list [ Direct; Tape; Coordinate; Padding ] in
  let* adt =
    of_list ~pp:Kernels_cases.pp_dtype
      D.[ Any Float32; Any Int32; Any Float16; Any Uint8; Any Int16 ]
  in
  let row = map (fun e -> [| e |]) extent in
  let* ashape =
    match ak with
    | Direct | Tape ->
        frequency
          [
            (4, row);
            (1, map (fun e -> [| 1; e |]) extent);
            (1, of_list [ [||]; [| 1; 1 |] ]);
          ]
    | Coordinate | Padding -> row
  in
  let r = Array.length ashape in
  let e = if r = 0 then 1 else ashape.(r - 1) in
  let* askew = int_range 0 3 in
  let* abcast = frequency [ (3, constant false); (1, constant true) ] in
  let* alo = int_range (-2) 3 in
  let* asrc = int_range 1 (e + 2) in
  let+ aseed = nat in
  { ak; adt; ashape; askew; abcast; alo; asrc; aseed }

let axis_prog c =
  let dt = c.adt in
  match c.ak with
  | Direct | Tape | Padding ->
      P.v ~ins:[| dt; dt |]
        [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
        ~outs:[| 2 |]
  | Coordinate ->
      let r = Array.length c.ashape in
      P.v ~ins:[| dt; dt |]
        [|
          P.In 0;
          In 1;
          Coord (r - 1);
          Op1 (Cast, dt, 2);
          Op2 (Binary Add, 0, 3);
          Op2 (Binary Add, 4, 1);
        |]
        ~outs:[| 5 |]

(* [c]'s call on [dev]: its destination and its loads, drawn from [c]'s seed
   alike on every device. The first load lies [askew] elements into its buffer,
   or is [c]'s padding of a row of [asrc]; the second is the loop's shape, or
   one element broadcast to it. *)
let axis_call dev c =
  let st = Random.State.make [| c.aseed |] in
  let (D.Any d as dt) = c.adt in
  let on (A.Any h) =
    let h = A.expect d (A.Any h) in
    if Rig.equal dev Rig.host then h else A.to_device dev h
  in
  let r = Array.length c.ashape in
  let first =
    match c.ak with
    | Padding ->
        let x = on (staged_fill st dt [| c.asrc |] ~raw:false) in
        let pad =
          {
            Nx_kernel.lo = [| c.alo |];
            hi = [| c.ashape.(0) - c.alo - c.asrc |];
            interior = [| 0 |];
            windows = [||];
          }
        in
        Nx_kernel.Padded { x; fill = D.of_float d 3.; pad }
    | (Direct | Tape | Coordinate) when r = 0 ->
        Nx_kernel.Plain (on (staged_fill st dt [||] ~raw:false))
    | Direct | Tape | Coordinate ->
        let grown = Array.copy c.ashape in
        grown.(r - 1) <- grown.(r - 1) + c.askew;
        let b = on (staged_fill st dt grown ~raw:false) in
        let all e = { M.start = 0; count = e; step = 1 } in
        let rs = Array.map all c.ashape in
        rs.(r - 1) <- { M.start = c.askew; count = c.ashape.(r - 1); step = 1 };
        Nx_kernel.Plain (Option.get (A.move (M.Slice (Iarray.of_array rs)) b))
  in
  let second =
    if c.abcast then
      let b = on (staged_fill st dt (Array.make r 1) ~raw:false) in
      Option.get (A.move (M.Broadcast (Iarray.of_array c.ashape)) b)
    else on (staged_fill st dt c.ashape ~raw:false)
  in
  let z = A.create dev d (Iarray.of_array c.ashape) in
  ([| A.Any z |], [| first; Nx_kernel.Plain second |])

let axis_run (module K : Nx_kernel.S) dev c =
  let dsts, loads = axis_call dev c in
  let direct = c.ak <> Tape in
  let a, traced =
    Fun.protect
      ~finally:(fun () -> Nx_metal.bound_direct true)
      (fun () ->
        Nx_metal.bound_direct direct;
        Plan.traced (fun () ->
            K.map (axis_prog c) ~dsts:(Iarray.of_array dsts)
              (Iarray.of_array loads)))
  in
  equal ~msg:(Format.asprintf "%a" pp_axis_map c) answer A.Done a;
  (dsts, traced)

(* nx_metal.mli: a map has nx.cpu's bits at any length, quads or not. *)
let law_axis_map c =
  let b = backend () in
  let want, _ = axis_run (module Nx_cpu) Rig.host c in
  let got, traced = axis_run (module Nx_metal) b.device c in
  let n = numel c.ashape in
  let odd = n mod 4 <> 0 && n > 1 in
  let quads =
    List.exists
      (fun t -> mode_of t land Nx_metal.Kernels.loop_quads <> 0)
      traced
  in
  let b32 = ran Loop_map_direct32 traced || ran Loop_map_direct32_flat traced in
  cover "a direct map's quads at a length of no multiple of 4"
    (c.ak = Direct && quads && odd);
  cover "the tape's quads at a length of no multiple of 4"
    (c.ak = Tape && quads && odd);
  cover "a coordinate's map at a length of no multiple of 4"
    (c.ak = Coordinate && odd);
  cover "a padding's quads at a length of no multiple of 4"
    (c.ak = Padding && quads && odd);
  cover "the tape's quads of one element" (c.ak = Tape && quads && n = 1);
  cover "the 32-bit kernel of one element" (b32 && n = 1);
  cover "the 32-bit kernel over no axis" (b32 && Array.length c.ashape = 0);
  cover "the 32-bit kernel at a length of no multiple of 4" (b32 && odd);
  cover "quads over a skewed load" (quads && odd && c.askew > 0);
  cover "quads over a broadcast load" (quads && odd && c.abcast);
  cover "quads past a group's elements" (quads && odd && n > 1024);
  same_bits want got

(* Each kind at a length of no multiple of 4, and the direct kernel and the tape
   over no axis and one element, which the draws can miss. *)
let axis_examples =
  let c ?(skew = 0) ?(lo = 0) ?(src = 1) ak dt shape =
    {
      ak;
      adt = dt;
      ashape = shape;
      askew = skew;
      abcast = false;
      alo = lo;
      asrc = src;
      aseed = 3;
    }
  in
  let f32 = D.Any D.Float32 and i32 = D.Any D.Int32 in
  [
    c Direct f32 [||];
    c Direct i32 [| 1 |];
    c Direct f32 [| 1; 1 |];
    c ~skew:1 Direct f32 [| 7 |];
    c Tape (D.Any D.Float16) [||];
    c Tape f32 [| 1 |];
    c Tape f32 [| 1; 5 |];
    c Coordinate f32 [| 7 |];
    c ~lo:1 ~src:5 Padding f32 [| 7 |];
  ]

(* Loop.kernel_of: a map's header names the kernel of its paths, and a header of
   tiles or of a direct map at a v, width or padded load that no kernel of them
   has raises; so does one of 32-bit lanes outside a direct map of quads whose
   form takes its elements and gives its result as they are, and one of 8-bit
   lanes outside a direct map of quads whose form takes its elements as they are
   or as truths, gives its result as it is or wrapped, and computes no long
   kind. *)
let test_kernel_of () =
  let module K = Nx_metal.Kernels in
  let module L = Nx_metal.Loop in
  (* The forms of a kind of two loads (its code aside), as they are; of one
     whose first load is decoded from float16; and of one whose result is
     encoded to float16. *)
  let add = (2 lsl K.direct_count) lor (0x10 lsl K.direct_operands) in
  let decoded = add lor (K.direct_f16 lsl (K.direct_operands + 2)) in
  let encoded = add lor (K.direct_f16 lsl K.direct_result) in
  (* And of a kind of two loads of one narrow float, each decoded and its result
     encoded, and of one that decodes a bfloat16 into a float16. *)
  let coded c =
    add
    lor (c lsl (K.direct_operands + 2))
    lor (c lsl (K.direct_operands + 6))
    lor (c lsl K.direct_result)
  in
  let bf16 = coded K.direct_bf16 and f16 = coded K.direct_f16 in
  let mixed =
    coded K.direct_bf16
    lxor (K.direct_bf16 lsl K.direct_result)
    lor (K.direct_f16 lsl K.direct_result)
  in
  let header ?(form = add) v flags =
    let h = Array.make (K.loop_held - K.loop_mode) 0 in
    h.(0) <- v lor flags;
    h.(K.loop_direct - K.loop_mode) <- form;
    h
  in
  let tiled = K.loop_tiled and padded = K.loop_padded and wide = K.loop_wide in
  let quads = K.loop_quads and b32 = K.loop_bits32 in
  let b16 = K.loop_bits16 and b8 = K.loop_bits8 in
  List.iter
    (fun (mode, v, flags, want) ->
      equal ~msg:want string want (K.name (L.kernel_of mode (header v flags))))
    [
      (L.Direct, 4, b32 lor quads, "loop_map_direct32");
      (L.Direct, 4, b32 lor quads lor tiled, "loop_map_direct32_tiled");
      (L.Direct, 4, quads, "loop_map_direct");
      (L.Direct, 4, quads lor tiled, "loop_map_direct_tiled");
      (L.Map, 4, 0, "loop_map_v4");
      (L.Map, 4, K.loop_quads, "loop_map_v4");
      (L.Map, 4, padded, "loop_map_padded");
      (L.Map, 4, padded lor K.loop_quads lor K.loop_coords, "loop_map_padded");
      (L.Map, 1, padded, "loop_map_v1");
      (L.Map, 1, wide lor padded, "loop_map_wide");
      (L.Map, 4, tiled, "loop_map_tiled");
      (L.Map, 4, tiled lor K.loop_quads, "loop_map_tiled");
      (L.Direct, 4, 0, "loop_map_direct");
      (L.Direct, 4, tiled, "loop_map_direct_tiled");
    ];
  List.iter
    (fun (mode, v, flags, msg) ->
      raises_match ~msg (Exn.invalid_arg ~sub:"Loop: no ") (fun () ->
          L.kernel_of mode (header v flags)))
    [
      (L.Map, 1, tiled, "tiles at 1 element a thread");
      (L.Map, 1, wide lor tiled, "tiles of 64-bit indices");
      (L.Map, 4, wide lor tiled, "tiles at 4 of 64-bit indices");
      (L.Map, 4, tiled lor padded, "tiles of a padded load");
      (L.Map, 4, wide, "a map at 4 of 64-bit indices");
      (L.Map, 2, 0, "a map at 2 elements a thread");
      (L.Direct, 1, 0, "a direct map at 1 element a thread");
      (L.Direct, 1, wide, "a direct map of 64-bit indices");
      (L.Direct, 4, padded, "a direct map of a padded load");
      (L.Direct, 4, tiled lor padded, "direct tiles of a padded load");
      (L.Direct, 1, tiled, "direct tiles at 1 element a thread");
      (L.Direct, 4, b32, "32-bit lanes without quads");
      (L.Direct, 4, b32 lor tiled, "32-bit tiles without quads");
      (L.Direct, 4, b32 lor quads lor wide, "32-bit lanes of 64-bit indices");
      (L.Direct, 1, b32 lor quads, "32-bit lanes at 1 element a thread");
      (L.Direct, 4, b32 lor quads lor padded, "32-bit lanes of a padded load");
      (L.Map, 4, b32 lor quads, "32-bit lanes in a tape's map");
      (L.Reduce, 4, b32, "32-bit lanes in a reduction");
      (L.Direct, 4, b8 lor quads lor wide, "8-bit lanes of 64-bit indices");
      (L.Direct, 4, b8 lor quads lor padded, "8-bit lanes of a padded load");
      (L.Map, 4, b8 lor quads, "8-bit lanes in a tape's map");
      (L.Reduce, 4, b8, "8-bit lanes in a reduction");
    ];
  (* A form whose slots are placed flat or at their first element runs a flat
     kernel, of its kind's cost. *)
  let placed form =
    form
    lor (K.direct_flat_on lsl K.direct_places)
    lor (K.direct_flat lsl (K.direct_places + 2))
    lor (K.direct_first lsl (K.direct_places + 4))
  in
  (* A float32 Exp of one load, a long kind: NX_OP1_EXP is 8 (nx_kernel.h). *)
  let long = (1 lsl K.direct_count) lor (K.loop_f32 lsl 7) lor 8 in
  (* Forms of bytes: two loads as they are, the first as a truth, the result
     wrapped into int8, and into int4, which a byte does not hold; and a uint32
     Idiv of two loads, a long kind (kernels.h, NX_DIRECT_LONG_INT2's bit 4). *)
  let truth = add lor (K.direct_truth lsl (K.direct_operands + 2)) in
  let wrap_into dt =
    add
    lor (K.direct_wrap lsl K.direct_result)
    lor (D.code dt lsl K.direct_wrap_dt)
  in
  let wrapped = wrap_into D.Int8 and wrapped4 = wrap_into D.Int4 in
  let bytes_long =
    (2 lsl K.direct_count) lor (K.loop_u32 lsl 7) lor (K.loop_binary + 4)
  in
  List.iter
    (fun (form, want) ->
      equal ~msg:want string want
        (K.name (L.kernel_of L.Direct (header ~form 4 (b32 lor quads)))))
    [
      (placed add, "loop_map_direct32_flat");
      (placed long, "loop_map_direct32_long_flat");
      (long, "loop_map_direct32_long");
    ];
  List.iter
    (fun (form, flags, msg) ->
      raises_match ~msg (Exn.invalid_arg ~sub:"Loop: no ") (fun () ->
          L.kernel_of L.Direct (header ~form 4 flags)))
    [
      (decoded, b32 lor quads, "32-bit lanes of a decoded load");
      (encoded, b32 lor quads, "32-bit lanes of an encoded result");
      (decoded, b32 lor quads lor tiled, "32-bit tiles of a decoded load");
      (bf16, b16, "16-bit lanes without quads");
      (bf16, b16 lor quads lor b32, "16- and 32-bit lanes");
      (add, b16 lor quads, "16-bit lanes of loads as they are");
      (decoded, b16 lor quads, "16-bit lanes of one decoded load");
      (encoded, b16 lor quads, "16-bit lanes of an encoded result alone");
      (mixed, b16 lor quads, "16-bit lanes of two codecs");
      (add, b8, "8-bit lanes without quads");
      (add, b8 lor tiled, "8-bit tiles without quads");
      (add, b8 lor quads lor b32, "8- and 32-bit lanes");
      (add, b8 lor quads lor b16, "8- and 16-bit lanes");
      (decoded, b8 lor quads, "8-bit lanes of a decoded load");
      (encoded, b8 lor quads, "8-bit lanes of an encoded result");
      (wrapped4, b8 lor quads, "8-bit lanes of a result wrapped into 4 bits");
      (bytes_long, b8 lor quads, "8-bit lanes of a long kind");
      (bytes_long, b8 lor quads lor tiled, "8-bit tiles of a long kind");
    ];
  List.iter
    (fun (form, flags, want) ->
      equal ~msg:want string want
        (K.name (L.kernel_of L.Direct (header ~form 4 flags))))
    [
      (bf16, b16 lor quads, "loop_map_direct16_bf16");
      (bf16, b16 lor quads lor tiled, "loop_map_direct16_bf16_tiled");
      (f16, b16 lor quads, "loop_map_direct16_f16");
      (f16, b16 lor quads lor tiled, "loop_map_direct16_f16_tiled");
      (bf16, quads, "loop_map_direct");
      (add, b8 lor quads, "loop_map_direct8");
      (add, b8 lor quads lor tiled, "loop_map_direct8_tiled");
      (truth, b8 lor quads, "loop_map_direct8");
      (wrapped, b8 lor quads lor tiled, "loop_map_direct8_tiled");
      (bytes_long, quads, "loop_map_direct");
    ]

(* The padded coordinate's extremes: padding by 2^31 - 1 before 4 elements and
   cropping as much after keeps 32 bits; by 2^31 the loop is wide; and a window
   of dilation 3 over an axis padded by 2^31 after, whose step leaves one
   window, reaches past 2^31 through the padded extent alone, so it is wide too,
   though its taps read elements 0 and 3. Each has nx.cpu's bits. *)
let test_padded_reach () =
  let b = backend () in
  let case ~lo ~hi ?window () =
    {
      pd_dt = D.Any D.Float32;
      pd_shape = [||];
      pd_pads =
        [|
          {
            Nx_kernel.lo = [| lo |];
            hi = [| hi |];
            interior = [| 0 |];
            windows =
              (match window with
              | None -> [||]
              | Some (step, dilation) ->
                  [| { M.axis = 0; size = 2; step; dilation } |]);
          };
        |];
      pd_srcs = [| [| 4 |] |];
      pd_rev = [| false |];
      pd_plain = false;
      pd_fill = 3.;
      pd_seed = 5;
    }
  in
  let edge = (1 lsl 31) - 1 in
  List.iter
    (fun (name, c, want) ->
      let ls = padded_launches c in
      equal ~msg:name (list string)
        [ Nx_metal.Kernels.name want ]
        (Array.to_list
           (Array.map
              (fun (l : Plan.launch) -> Nx_metal.Kernels.name l.kernel)
              ls));
      same_bits
        (padded_run (module Nx_cpu) Rig.host c)
        (padded_run (module Nx_metal) b.device c))
    [
      ( "lo of 2^31 - 1",
        case ~lo:edge ~hi:(-edge) (),
        Nx_metal.Kernels.Loop_map_padded );
      ( "lo of 2^31",
        case ~lo:(edge + 1) ~hi:(-edge - 1) (),
        Nx_metal.Kernels.Loop_map_wide );
      ( "a window past 2^31",
        case ~lo:0 ~hi:(edge + 1) ~window:((1 lsl 31) + 8, 3) (),
        Nx_metal.Kernels.Loop_map_wide );
    ]

(* A tiled map of bytes, of 200 rows: 7 tiles along b, in runs of 4, so that a
   plane's last run holds a group without a tile. In 3 planes, cut into launches
   of a few groups each, such groups lie in launches that start past group 0;
   its bits are nx.cpu's. *)
let test_staged_launches () =
  let b = backend () in
  let c =
    {
      st_x = D.Any D.Uint8;
      st_y = D.Any D.Uint8;
      st_batch = 3;
      st_rows = 200;
      st_cols = 300;
      st_pitch = 200;
      st_tx = false;
      st_ty = true;
      st_neg = false;
      st_skip = 0;
      st_extra = `None;
      st_raw = false;
      st_alias = false;
      st_mod = false;
      st_seed = 11;
    }
  in
  bounded 30e-6 (fun () ->
      let dsts, loads = staged_call Rig.host c in
      let launches =
        match loop_map ~holds:(fun _ -> 1024) (staged_prog c) ~dsts loads with
        | Ok (Some (p : Plan.t)) -> Array.length p.launches
        | Ok None | Error _ -> fail "no plan"
      in
      at_least int ~than:3 launches;
      equal ~msg:"staged" bool true (case_shared c >= stage);
      same_bits
        (staged_run (module Nx_cpu) Rig.host c)
        (staged_run (module Nx_metal) b.device c))

(* nx_metal.mli, Time: a tiled map charges its operands their lines along a and
   its staged loads' along b. The host plans of the bench's t_add rows, z = x +
   t(y) into a contiguous z, with the times their charges count past the bytes
   they move. The estimates' milliseconds are baselines. *)
let test_tile_estimates () =
  let estimate (D.Any d) n =
    let bytes = n * n * D.bytes d 1 in
    let buffer = Rig.Buffer.create Rig.host (3 * bytes) in
    let shape = Iarray.of_list [ n; n ] in
    let at k strides =
      A.Any
        (A.v d
           (L.v ~offset:(k * n * n) ~strides:(Iarray.of_list strides) shape)
           buffer)
    in
    let (A.Any x) = at 0 [ n; 1 ] in
    let (A.Any y) = at 1 [ 1; n ] in
    let add =
      P.v ~ins:[| D.Any d; D.Any d |]
        [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
        ~outs:[| 2 |]
    in
    match
      loop_map
        ~holds:(fun _ -> 1024)
        add
        ~dsts:[| at 2 [ n; 1 ] |]
        [| Plain x; Plain y |]
    with
    | Ok (Some (p : Plan.t)) ->
        let s =
          Array.fold_left
            (fun t (l : Plan.launch) ->
              t
              +. Plan.seconds
                   ~threads:(Nx_metal.Kernels.threads l.kernel)
                   ~groups:l.groups ~resident:l.resident l.cost)
            0. p.launches
        in
        let charged =
          Array.fold_left
            (fun t (l : Plan.launch) -> t + l.cost.bytes)
            0 p.launches
        in
        Printf.sprintf "%s %s %dx%d: %d, %d, %.2fx, %.2f ms"
          (Nx_metal.Kernels.name p.launches.(0).kernel)
          (D.name d) n n (Array.length p.launches) p.launches.(0).shared
          (Float.of_int charged /. Float.of_int (3 * bytes))
          (s *. 1e3)
    | Ok None -> "no plan"
    | Error _ -> "declined"
  in
  (* Each row's kernel, dtype and shape; its launches, a group's threadgroup
     bytes, its charge over the bytes it moves, and its estimate. *)
  let rows =
    [
      estimate (D.Any D.Float32) 1024;
      estimate (D.Any D.Float32) 8192;
      estimate (D.Any D.Float16) 8192;
      estimate (D.Any D.Uint8) 8192;
    ]
  in
  expect (String.concat "\n" rows)
  @@ __POS_OF__
       {|
    loop_map_direct32_tiled float32 1024x1024: 1, 4224, 2.00x, 1.15 ms
    loop_map_direct32_tiled float32 8192x8192: 2, 4224, 2.00x, 72.91 ms
    loop_map_direct16_f16_tiled float16 8192x8192: 2, 4224, 4.00x, 93.99 ms
    loop_map_direct8_tiled uint8 8192x8192: 2, 4224, 8.00x, 79.92 ms
    |}

(* A loop over one operand of [lshape] at strides [lstrides] (0 at an extent of
   1), [lskip] elements past its least position's minimum: a map of x + x, or of
   x plus its last coordinate where [lcoord], whose walk is then never merged; a
   Sum along its last axis where [lsum]. *)
type lines_case = {
  ldt : D.any;
  lshape : int array;
  lstrides : int array;
  lskip : int;
  lcoord : bool;
  lsum : bool;
}

let pp_lines_case ppf c =
  let ints a = String.concat ";" (List.map string_of_int (Array.to_list a)) in
  Format.fprintf ppf "%a [%s] at [%s] +%d%s%s" Kernels_cases.pp_dtype c.ldt
    (ints c.lshape) (ints c.lstrides) c.lskip
    (if c.lcoord then " coord" else "")
    (if c.lsum then " sum" else " map")

let lines_cases =
  let open Gen in
  with_pp pp_lines_case
  @@ let* ldt =
       frequency
         [ (3, constant (D.Any D.Float32)); (2, constant (D.Any D.Uint8)) ]
     in
     let* r = int_range 1 3 in
     let* lshape =
       such_that
         (fun a -> numel a <= 1 lsl 14)
         (array ~size:(constant r) (of_list [ 1; 2; 3; 4; 5; 8; 17; 64; 300 ]))
     in
     (* Each axis' stride from the inside out: a broadcast, a short step, a line
        or more, or one that chains with the axis inside it. *)
     let step = of_list [ `Zero; `One; `Two; `Three; `Seven; `Line; `Chain ] in
     let* steps = array ~size:(constant r) step in
     let* signs = array ~size:(constant r) bool in
     let* lskip = int_range 0 40 in
     let* lcoord = bool in
     let+ lsum = bool in
     let lstrides = Array.make r 0 in
     let inner = ref 1 in
     for a = r - 1 downto 0 do
       let m =
         match steps.(a) with
         | `Zero -> 0
         | `One -> 1
         | `Two -> 2
         | `Three -> 3
         | `Seven -> 7
         | `Line -> 64
         | `Chain -> !inner
       in
       let st = if lshape.(a) = 1 then 0 else if signs.(a) then -m else m in
       lstrides.(a) <- st;
       if st <> 0 then inner := abs st * lshape.(a)
     done;
     {
       ldt;
       lshape;
       lstrides;
       lskip;
       lcoord;
       lsum = lsum && ldt = D.Any D.Float32;
     }

let lines_examples =
  let c ?(dt = D.Any D.Float32) ?(coord = false) ?(sum = false) shape strides =
    {
      ldt = dt;
      lshape = shape;
      lstrides = strides;
      lskip = 0;
      lcoord = coord;
      lsum = sum;
    }
  in
  [
    (* A column broadcast along rows of 2. *)
    c [| 4096; 2 |] [| 64; 0 |];
    (* A column, its extent-1 axis kept by a coordinate's read. *)
    c ~coord:true [| 4096; 1 |] [| 64; 0 |];
    (* Rows reversed. *)
    c [| 2048; 4 |] [| -64; 1 |];
    (* 2x2 blocks of rows of 64. *)
    c [| 1024; 2; 2 |] [| 64; 4; 1 |];
    (* 3 channels of 4, of bytes. *)
    c ~dt:(D.Any D.Uint8) [| 4096; 3 |] [| 4; 1 |];
    (* Windows of 32 that overlap. *)
    c [| 300; 32 |] [| 1; 1 |];
    (* Transposed reads a map stages: of floats, of bytes, batched. *)
    c [| 64; 64 |] [| 1; 64 |];
    c ~dt:(D.Any D.Uint8) [| 64; 64 |] [| 1; 64 |];
    c [| 3; 64; 33 |] [| 2112; 1; 64 |];
    (* Sums along short rows, merged and not. *)
    c ~sum:true [| 4096; 2 |] [| 64; 1 |];
    c ~sum:true ~coord:true [| 2048; 5 |] [| -64; 1 |];
    (* A sum of one term into outputs whose first axis, of extent 1, has stride
       0: the walk drops it. *)
    c ~sum:true [| 1; 300; 1 |] [| 0; 0; 0 |];
  ]

(* The distinct positions of [a]'s elements. *)
let positions (type a b) (a : (a, b) A.t) =
  let l = A.layout a in
  let shape = Iarray.to_array (L.shape l) in
  let strides = Iarray.to_array (L.strides l) in
  let seen = Hashtbl.create 4096 in
  let rec go k pos =
    if k = Array.length shape then Hashtbl.replace seen pos ()
    else
      for i = 0 to shape.(k) - 1 do
        go (k + 1) (pos + (i * strides.(k)))
      done
  in
  go 0 (L.offset l);
  Hashtbl.length seen

let law_lines c =
  let (D.Any d) = c.ldt in
  let w = D.bytes d 1 and r = Array.length c.lshape in
  let reach sign =
    Array.fold_left ( + ) 0
      (Array.mapi
         (fun a st -> (c.lshape.(a) - 1) * Int.max 0 (sign * st))
         c.lstrides)
  in
  let offset = c.lskip + reach (-1) in
  let need = offset + reach 1 + 1 in
  let outs = if c.lsum then Array.sub c.lshape 0 (r - 1) else c.lshape in
  let buffer = Rig.Buffer.create Rig.host ((need + numel outs + 2) * 4) in
  let x =
    A.v d
      (L.v ~offset
         ~strides:(Iarray.of_array c.lstrides)
         (Iarray.of_array c.lshape))
      buffer
  in
  let carrier = if c.lsum then D.Any D.Float32 else c.ldt in
  let (D.Any e) = carrier in
  let z =
    A.v e
      (L.v
         ~offset:((need * w / D.bytes e 1) + 1)
         ~strides:(L.strides (L.contiguous (Iarray.of_array outs)))
         (Iarray.of_array outs))
      buffer
  in
  let p =
    if c.lcoord then
      P.v ~ins:[| c.ldt |]
        [|
          P.In 0; Coord (r - 1); Op1 (Cast, c.ldt, 1); Op2 (Binary Add, 0, 2);
        |]
        ~outs:[| 3 |]
    else P.v ~ins:[| c.ldt |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |]
  in
  let planned =
    if c.lsum then
      loop_reduce
        ~holds:(fun _ -> 1024)
        p ~shape:(Iarray.of_array c.lshape)
        ~axes:[| r - 1 |]
        [| (Nx_kernel.Monoid Sum, 0) |]
        ~dsts:[| A.Any z |] [| Plain x |]
    else loop_map ~holds:(fun _ -> 1024) p ~dsts:[| A.Any z |] [| Plain x |]
  in
  let charged =
    match planned with
    | Ok (Some (p : Plan.t)) ->
        Array.fold_left
          (fun t (l : Plan.launch) -> t + l.cost.bytes)
          0 p.launches
    | Ok None -> fail "no plan"
    | Error _ -> fail "declined"
  in
  let last = r - 1 in
  let short st = st <> 0 && abs st * w < Plan.line in
  let stepping = Array.exists (fun st -> st <> 0) c.lstrides in
  cover "a broadcast innermost axis"
    (c.lstrides.(last) = 0 && c.lshape.(last) > 1);
  cover "an innermost extent of 1 in an unmerged walk"
    (c.lshape.(last) = 1 && c.lcoord);
  cover "a negative stride" (Array.exists (fun st -> st < 0) c.lstrides);
  cover "several short axes"
    (List.length (List.filter short (Array.to_list c.lstrides)) >= 2);
  cover "1-byte elements" (w = 1);
  cover "elements that overlap"
    (stepping
    && Array.for_all2 (fun st n -> st <> 0 || n = 1) c.lstrides c.lshape
    && positions x < numel c.lshape);
  cover "an unmerged walk" c.lcoord;
  cover "a merged walk" (not c.lcoord);
  cover "a sum" c.lsum;
  cover "a map" (not c.lsum);
  at_least int ~than:((touched x + touched z) * Plan.line) charged

(* A drawn scan of a class, of 1 to 32 axes, mostly of 2, its terms near or
   below the most its elements allow. *)
type kept_scan = {
  kc : int;
  kshape : int array;
  kstep : bool;
  kholds : int;
  near : bool;  (** Within a twentieth of the most terms it keeps. *)
}

let pp_kept_scan ppf k =
  Format.fprintf ppf "%s %a%s, %d threads" (List.nth long_cases k.kc).what
    pp_shape k.kshape
    (if k.kstep then " stepping" else "")
    k.kholds

let kept_scan =
  let open Gen in
  let* kc = int_range 0 (List.length long_cases - 1) in
  let* rank = int_range 1 32 in
  let* kstep = bool in
  let* kholds = of_list [ 256; 576; 704; 1024; 3072 ] in
  let* others =
    array
      ~size:(constant (rank - 1))
      (frequency [ (6, constant 2); (1, constant 1); (1, constant 3) ])
  in
  let* place = int_range 0 19 in
  let c = List.nth long_cases kc in
  let across = Array.fold_left ( * ) 1 others in
  let most = kept_elements (depends ~carried:c.carried c.p) / across in
  (* Half the draws within a twentieth below the most terms kept, the others
     spread below it. *)
  let terms =
    Int.max 1
      (if place < 10 then most - (most * place / 200)
       else most * (place - 9) / 11)
  in
  let near = most > 0 && terms * 20 >= most * 19 in
  let kshape = Array.append [| terms |] others in
  with_pp pp_kept_scan (constant { kc; kshape; kstep; kholds; near })

(* nx_metal.mli: a scan declines only where one element's program passes a
   launch, whatever threads Metal lets a group have: drawn within
   [kept_elements]. *)
let law_kept_scan k =
  let c = List.nth long_cases k.kc in
  let elements = Array.fold_left ( * ) 1 k.kshape in
  let axes =
    Array.fold_left (fun n e -> if e > 1 then n + 1 else n) 0 k.kshape
  in
  let kept = elements <= kept_elements (depends ~carried:c.carried c.p) in
  cover "within a twentieth of the most terms kept" (k.near && kept);
  cover "past one chunk a slice" (kept && k.kshape.(0) > 4096);
  cover "past 16 axes of more than one element" (kept && axes > 16);
  cover "stepping by one" k.kstep;
  cover "from a carry" c.carried;
  if kept then begin
    let buffer = Lazy.force long_buffer in
    match
      long_plan ~holds:k.kholds ~carried:c.carried buffer c.p c.r ~into:c.into
        ~shape:k.kshape ~step:k.kstep
    with
    | Ok _ -> ()
    | Error a -> fail_declined c ~step:k.kstep ~holds:k.kholds k.kshape a
  end

(* A scan of a class over [grank] axes, along its first, its [2^gslices] slices
   spread over as many axes after it as the rank holds, each of 2 but the last,
   past 4096 terms a slice, at [gholds] threads a group; its results step by one
   along the slice where [gstep]. *)
type grown = {
  gc : int;
  grank : int;
  gslices : int;
  gholds : int;
  gstep : bool;
}

let grown_shape g ~terms =
  let apart = Int.min (g.grank - 1) g.gslices in
  Array.init g.grank (fun i ->
      if i = 0 then terms
      else if i < apart then 2
      else if i = apart then 1 lsl (g.gslices - apart + 1)
      else 1)

let pp_grown ppf g =
  let others = Array.sub (grown_shape g ~terms:1) 1 (g.grank - 1) in
  Format.fprintf ppf "%s along the first of n x %a%s, %d threads"
    (List.nth long_cases g.gc).what pp_shape others
    (if g.gstep then " stepping" else "")
    g.gholds

let grown =
  let open Gen in
  let* gc = int_range 0 (List.length long_cases - 1) in
  let* grank = of_list [ 8; 20; 32 ] in
  let* gslices = int_range 8 18 in
  let* gholds = of_list [ 256; 576; 704; 1024; 3072 ] in
  let* gstep = frequency [ (3, constant true); (1, constant false) ] in
  with_pp pp_grown (constant { gc; grank; gslices; gholds; gstep })

(* nx_metal.mli: a scan whose element's program fits a launch plans at every
   length, a call past a part's seconds in several parts: at 1 term a slice, one
   chunk of 4096 and one more, a block's switch, and [kept_elements], a few
   seconds by the estimate, under parts of 1 s. *)
let law_grown_plans g =
  let c = List.nth long_cases g.gc in
  let buffer = Lazy.force long_buffer in
  let plan terms =
    S.parted ~seconds:1. (fun () ->
        long_plan ~holds:g.gholds ~carried:c.carried buffer c.p c.r ~into:c.into
          ~shape:(grown_shape g ~terms) ~step:g.gstep)
  in
  let planned terms = Result.is_ok (plan terms) in
  (* A Logsumexp or Moments slice takes one pass up to its block's terms and
     three past them: the block's terms, its network launches' NX_LOOP_SEGMENT
     at 4096 terms a slice, and one more. *)
  let switch =
    let network (l : Plan.launch) =
      match l.kernel with
      | Loop_network_v4 | Loop_network_v1 | Loop_network_wide -> true
      | _ -> false
    in
    match plan 4096 with
    | Ok (Some p) -> (
        match List.find_opt network (Array.to_list p.launches) with
        | Some l ->
            let b =
              Int64.to_int
                (Bytes.get_int64_le l.words (8 * Nx_metal.Kernels.loop_segment))
            in
            [ b; b + 1 ]
        | None -> [])
    | Ok None | Error _ -> []
  in
  (* A plan's time grows with its launches: no length past [kept_elements]. *)
  let slices = 1 lsl g.gslices in
  let longest = kept_elements (depends ~carried:c.carried c.p) / slices in
  let lengths =
    List.filter
      (fun t -> t >= 1 && t <= longest)
      (List.sort_uniq compare ([ 1; 4096; 4097; longest ] @ switch))
  in
  List.iter
    (fun terms ->
      if not (planned terms) then
        failf "%a: declined at %d terms a slice" pp_grown g terms)
    lengths;
  cover "several parts at the longest"
    (longest >= 1
    &&
    match plan longest with
    | Ok (Some p) -> Array.length p.parts > 1
    | Ok None | Error _ -> false);
  cover "a Logsumexp or Moments"
    (match c.r with Logsumexp | Moments -> true | Monoid _ | Arg _ -> false);
  cover "a program past 64 nodes" (depends c.p > 64);
  cover "a Logsumexp or Moments at its block's terms and one more" (switch <> []);
  List.iter
    (fun rank -> cover (Printf.sprintf "%d axes" rank) (g.grank = rank))
    [ 8; 20; 32 ]

(* A scan on the GPU: class [sc] of [split_cases], along axis 1 of [sshape], in
   launches of at most [sbudget] seconds, its operands drawn from [sseed]. *)
type split_scan = { sc : int; sshape : int array; sbudget : float; sseed : int }

let split_cases =
  let f32 = D.Any D.Float32 and i4 = D.Any D.Int4 in
  let c64 = D.Any D.Complex64 in
  let case ?(carried = false) what p r into = { what; p; r; into; carried } in
  Nx_kernel.
    [
      case "a float32 sum" (identity f32) (Monoid Sum) f32;
      case "an Arg of float32" (identity f32) (Arg Max) f32;
      case "a sum of a float32 Exp" (exps f32 1) (Monoid Sum) f32;
      case "a sum of 4 float32 Exp-Logs" (exp_logs f32 4) (Monoid Sum) f32;
      case ~carried:true "a float32 sum from a carry" (identity f32)
        (Monoid Sum) f32;
      case "an Arg of int4" (identity i4) (Arg Min) i4;
      case "an int4 sum into int4" (identity i4) (Monoid Sum) i4;
      case "an Arg of 4 float32 Exp-Logs" (exp_logs f32 4) (Arg Max) f32;
      case "an Arg of a complex64 Pow" (self_pow c64) (Arg Max) c64;
    ]

let split_class = class_in split_cases

let pp_split_scan ppf s =
  Format.fprintf ppf "%s along axis 1 of %a, %g s a launch"
    (List.nth split_cases s.sc).what pp_shape s.sshape s.sbudget

(* The launch budget above which a reduction's group of a chunk's terms fits a
   launch, and a chunk's chain, stepping by one or not, and the carries of a
   slice of 2^22 terms do not; and one below which the group's runs take several
   launches. *)
let split_budget = 2e-3
let pieces_budget = 3e-4

(* Of the shorter draws, a third in the library's budget, where a strided chunk
   of a class that computes more than a load already splits, and a third in
   [pieces_budget]; a quarter of all at a slice of 2^22 terms in [split_budget],
   whose 1024 chunks' carries split. *)
let split_scan =
  let open Gen in
  let* sc = int_range 0 (List.length split_cases - 1) in
  let* long = frequency [ (1, constant true); (3, constant false) ] in
  let* before = if long then constant 1 else int_range 1 2 in
  let* after = if long then constant 1 else of_list [ 1; 3 ] in
  let* terms =
    if long then constant (1 lsl 22)
    else
      frequency
        [
          (2, int_range 1 64);
          (2, int_range 4095 4097);
          (4, int_range 4098 20_000);
          (1, constant (1 lsl 20));
        ]
  in
  let* sbudget =
    if long then constant split_budget
    else of_list [ Plan.budget; split_budget; pieces_budget ]
  in
  let* sseed = int in
  with_pp pp_split_scan
    (constant { sc; sshape = [| before; terms; after |]; sbudget; sseed })

(* A host array of [d] and [shape], from [st]: floats and complex parts in [-1,
   1], else any bits. *)
let split_operand st (D.Any d) shape =
  let x = A.create Rig.host d (Iarray.of_array shape) in
  let b = A.buffer x in
  let n = Rig.Buffer.length b in
  (match d with
  | D.Float32 | D.Complex64 ->
      let s = Bytes.create n in
      for i = 0 to (n / 4) - 1 do
        let f = Random.State.float st 2. -. 1. in
        Bytes.set_int32_le s (4 * i) (Int32.bits_of_float f)
      done;
      Rig.Buffer.blit_from_string (Bytes.to_string s) 0 b 0 n
  | _ -> Rig.Buffer.blit_from_string (random_bytes st n) 0 b 0 n);
  A.Any x

let scan_launch (t : Plan.traced) =
  match t.kernel with
  | Loop_scan_v4 | Loop_scan_v1 | Loop_scan_wide -> true
  | _ -> false

(* The word [i] of a launch's parameters. *)
let word (t : Plan.traced) i = Int64.to_int (Bytes.get_int64_le t.words (8 * i))

(* The threads a scan launch's chunk spreads its terms over, its header's
   NX_LOOP_GROUP. *)
let spread_of t = word t Nx_metal.Kernels.loop_group

(* A chunk spread over 256 threads, its results stepping by one and not, and
   spread past 16 for an Arg folded in words and one folded held; and a scan
   from a carry, one class of nine, which a few draws can miss. A slice of 2^22
   terms in [split_budget]: its chunks' totals in groups over several launches
   and its carries in several, and a plain sum's totals by a row's kernel. *)
let split_scan_examples =
  List.map
    (fun (what, sshape) ->
      { sc = split_class what; sshape; sbudget = Plan.budget; sseed = 4 })
    [
      ("a sum of 4 float32 Exp-Logs", [| 1; 5000; 3 |]);
      ("a sum of 4 float32 Exp-Logs", [| 2; 5000; 1 |]);
      ("an Arg of 4 float32 Exp-Logs", [| 1; 4500; 3 |]);
      ("an Arg of a complex64 Pow", [| 1; 4500; 3 |]);
      ("a float32 sum from a carry", [| 2; 5000; 1 |]);
    ]
  @ List.map
      (fun what ->
        {
          sc = split_class what;
          sshape = [| 1; 1 lsl 22; 1 |];
          sbudget = split_budget;
          sseed = 6;
        })
      [ "a sum of 4 float32 Exp-Logs"; "a float32 sum" ]

(* nx_metal.mli: a chunk's chain and a slice's carries run across as many
   launches as their time takes, with nx.cpu's bits. *)
let law_split_scan s =
  let b = backend () in
  let c = List.nth split_cases s.sc in
  let st = Random.State.make [| s.sseed |] in
  let (D.Any d) = Iarray.get c.p.P.ins 0 in
  let x = split_operand st (D.Any d) s.sshape in
  let carrier = Nx_kernel.carrier c.into in
  let carry =
    if c.carried then
      Some (split_operand st carrier [| s.sshape.(0); 1; s.sshape.(2) |])
    else None
  in
  let dts =
    match c.r with
    | Arg _ -> [ c.into; D.Any D.Int64 ]
    | Moments -> [ c.into; c.into ]
    | Logsumexp | Monoid _ -> [ c.into ]
  in
  let run (module K : Nx_kernel.S) dev ~on =
    let dsts =
      Array.of_list
        (List.map
           (fun (D.Any e) -> A.Any (A.create dev e (Iarray.of_array s.sshape)))
           dts)
    in
    let (A.Any x) = on x in
    let a =
      K.scan ?carry:(Option.map on carry) c.p ~axis:1 (c.r, 0)
        ~dsts:(Iarray.of_array dsts) [| Nx_kernel.Plain x |]
    in
    equal ~msg:(Format.asprintf "%a" pp_split_scan s) answer A.Done a;
    dsts
  in
  let want = run (module Nx_cpu) Rig.host ~on:Fun.id in
  let got, launches =
    bounded s.sbudget (fun () ->
        Plan.traced (fun () ->
            run
              (module Nx_metal)
              b.device
              ~on:(fun (A.Any a) -> A.Any (Support.on b a))))
  in
  let count f = List.length (List.filter f launches) in
  let totals =
    List.fold_left
      (fun n (t : Plan.traced) ->
        if is_reduce (Nx_metal.Kernels.name t.kernel) then
          let g, _, _ = t.groups in
          n + g
        else n)
      0 launches
  in
  let units = s.sshape.(0) * s.sshape.(2) * ((s.sshape.(1) + 4095) / 4096) in
  let spread =
    List.fold_left
      (fun m t -> if scan_launch t then Int.max m (spread_of t) else m)
      0 launches
  in
  let arg = match c.r with Arg _ -> true | _ -> false in
  cover "a chunk's chain in several launches" (count scan_launch > 1);
  cover "a group of a chunk's totals in several launches"
    (totals > (units + 3) / 4);
  cover "a slice's carries in several launches"
    (count (fun t -> t.kernel = Loop_carries) > 1);
  cover "past one chunk a slice" (s.sshape.(1) > 4096);
  cover "stepping by one" (s.sshape.(2) = 1);
  cover "from a carry" c.carried;
  cover "a slice's chunk totals by a row's kernel"
    (List.exists
       (fun (t : Plan.traced) ->
         String.starts_with ~prefix:"loop_row_" (Nx_metal.Kernels.name t.kernel))
       launches);
  cover "256 threads a chunk, not stepping" (spread = 256 && s.sshape.(2) > 1);
  cover "256 threads a chunk, stepping by one" (spread = 256 && s.sshape.(2) = 1);
  cover "an Arg spread past 16" (arg && spread > 16);
  cover "a held fold spread past 16"
    (Nx_kernel.carrier c.into = D.Any D.Complex64 && spread > 16);
  same_bits want got

(* A Logsumexp or Moments scan on the GPU: class [nc] of [net_cases], along axis
   1 of [nshape], in launches of at most [nbudget] seconds, its operand drawn
   from [nseed]. *)
type net_scan = { nc : int; nshape : int array; nbudget : float; nseed : int }

let net_cases =
  let f32 = D.Any D.Float32 and bf16 = D.Any D.Bfloat16 in
  let case what p r into = { what; p; r; into; carried = false } in
  Nx_kernel.
    [
      case "a float32 Logsumexp" (identity f32) Logsumexp f32;
      case "float32 Moments" (identity f32) Moments f32;
      case "bfloat16 Moments" (identity bf16) Moments bf16;
      case "a Logsumexp of a float32 Exp" (exps f32 1) Logsumexp f32;
    ]

(* nx_metal.mli: a group takes [W] slices side by side, a power of two: as many
   as its tiles hold a slice's terms in, at least 32 where the results do not
   step by one along the slice, at most the slices; a tile is [256 v / W] terms
   of each. *)
let net_side shape =
  let slices = shape.(0) * shape.(2) and terms = shape.(1) in
  let rec up w = if w >= Int.min terms 256 then w else up (2 * w) in
  let rec down w n = if 2 * w > n then w else down (2 * w) n in
  let floor = if shape.(2) = 1 then 1 else 32 in
  down 1 (Int.min slices (Int.max (256 / up 1) floor))

let pp_net_scan ppf s =
  Format.fprintf ppf "%s along axis 1 of %a, %g s a launch"
    (List.nth net_cases s.nc).what pp_shape s.nshape s.nbudget

(* Slices of [t] tiles and [r] terms more at one element a thread, [t] odd and
   at least 3 in two thirds of the draws, so that a tile's last term joins a
   network of tile blocks whose association a running fold does not give; to 65
   tiles, so that the tiles' network reaches 7 levels. *)
let tiled_slices nc =
  let open Gen in
  let* before = int_range 1 2 in
  let* after = of_list [ 1; 1; 3; 16 ] in
  let w = net_side [| before; 256; after |] in
  let tile = 256 / w in
  let* t =
    frequency
      [
        (4, of_list [ 3; 5; 7; 9; 17; 33; 65 ]);
        (2, of_list [ 0; 1; 2; 4; 8; 16; 64 ]);
      ]
  in
  let* r =
    frequency [ (1, constant 0); (1, constant 1); (2, int_range 1 (tile - 1)) ]
  in
  let terms = Int.max 1 ((t * tile) + r) in
  let* nbudget = of_list [ Plan.budget; 1e-4 ] in
  let* nseed = int in
  constant { nc; nshape = [| before; terms; after |]; nbudget; nseed }

(* 64 to 256 slices of 1 to 4 terms, whose groups take 64 to 256 slices side by
   side, their count a power of two or not, the results stepping by one or
   not. *)
let short_slices nc =
  let open Gen in
  let* terms = int_range 1 4 in
  let* many = of_list [ 64; 100; 128; 200; 255; 256 ] in
  let* stepping = bool in
  let* few = int_range 1 2 in
  let before, after = if stepping then (many, 1) else (few, many) in
  let* nbudget = of_list [ Plan.budget; 1e-4 ] in
  let* nseed = int in
  constant { nc; nshape = [| before; terms; after |]; nbudget; nseed }

(* Slices of 1 to 7 blocks of 4096 terms and a part of one, side by side or not,
   so that a block's last tile ends at the chain's result, which differs from
   the block's prefix joined with its total where 4 divides the block's number
   plus one, from 4 whole blocks; in launches of the budget, or of 0.1 ms, under
   which a block takes fewer tiles and the blocks' chain several launches. *)
let blocked_slices nc =
  let open Gen in
  let* before = int_range 1 2 in
  let* after = of_list [ 1; 3; 16 ] in
  let* blocks = of_list [ 1; 2; 3; 4; 5; 7 ] in
  let* part = of_list [ 0; 1; 2048 ] in
  let terms = Int.max 4097 ((blocks * 4096) + part) in
  let* nbudget = of_list [ Plan.budget; 1e-4 ] in
  let* nseed = int in
  constant { nc; nshape = [| before; terms; after |]; nbudget; nseed }

let net_scan =
  let open Gen in
  let* nc = int_range 0 (List.length net_cases - 1) in
  with_pp pp_net_scan
    (frequency
       [ (2, tiled_slices nc); (1, short_slices nc); (2, blocked_slices nc) ])

(* A host operand of [d]: floats in [-1, 1]; in a quarter of the draws, 1 to 3
   of them NaN, an infinity, -0 or a large magnitude, so that slices keep finite
   results past them; bfloat16's their top bits. *)
let net_operand st (D.Any d) shape =
  let x = A.create Rig.host d (Iarray.of_array shape) in
  let b = A.buffer x in
  let n = Array.fold_left ( * ) 1 shape in
  let values = Array.init n (fun _ -> Random.State.float st 2. -. 1.) in
  if Random.State.int st 4 = 0 then
    for _ = 1 to 1 + Random.State.int st 3 do
      values.(Random.State.int st n) <-
        (match Random.State.int st 5 with
        | 0 -> Float.nan
        | 1 -> Float.infinity
        | 2 -> Float.neg_infinity
        | 3 -> -0.
        | _ -> 80. *. (Random.State.float st 2. -. 1.))
    done;
  (match d with
  | D.Float32 ->
      let s = Bytes.create (4 * n) in
      Array.iteri
        (fun i f -> Bytes.set_int32_le s (4 * i) (Int32.bits_of_float f))
        values;
      Rig.Buffer.blit_from_string (Bytes.to_string s) 0 b 0 (4 * n)
  | D.Bfloat16 ->
      let s = Bytes.create (2 * n) in
      Array.iteri
        (fun i f ->
          let top = Int32.shift_right_logical (Int32.bits_of_float f) 16 in
          Bytes.set_uint16_le s (2 * i) (Int32.to_int top))
        values;
      Rig.Buffer.blit_from_string (Bytes.to_string s) 0 b 0 (2 * n)
  | _ -> invalid_arg "net_operand");
  A.Any x

let network_launch (t : Plan.traced) =
  match t.kernel with
  | Loop_network_v4 | Loop_network_v1 | Loop_network_wide -> true
  | _ -> false

(* A scan by [c] along axis 1 of [shape] of the host operand [x]: nx.cpu's
   results, and nx.metal's under [budget] with its launches. *)
let net_results ?(budget = Plan.budget) (b : Support.backend) c shape x =
  let dts =
    match c.r with
    | Moments -> [ c.into; c.into ]
    | Logsumexp | Monoid _ | Arg _ -> [ c.into ]
  in
  let run (module K : Nx_kernel.S) dev ~on =
    let dsts =
      Array.of_list
        (List.map
           (fun (D.Any e) -> A.Any (A.create dev e (Iarray.of_array shape)))
           dts)
    in
    let (A.Any x) = on x in
    let a =
      K.scan c.p ~axis:1 (c.r, 0) ~dsts:(Iarray.of_array dsts)
        [| Nx_kernel.Plain x |]
    in
    equal ~msg:(Printf.sprintf "%s: answer" c.what) answer A.Done a;
    dsts
  in
  let want = run (module Nx_cpu) Rig.host ~on:Fun.id in
  let got, launches =
    bounded budget (fun () ->
        Plan.traced (fun () ->
            run
              (module Nx_metal)
              b.device
              ~on:(fun (A.Any a) -> A.Any (Support.on b a))))
  in
  (want, got, launches)

let same_bits want got = same_bits want got

(* nx_metal.mli: a Logsumexp or Moments scan has nx.cpu's bits, by
   nx_kernel.mli's network, however its tiles fall and its launches split. *)
let law_network_scan s =
  let b = backend () in
  let c = List.nth net_cases s.nc in
  let st = Random.State.make [| s.nseed |] in
  let (D.Any d) = Iarray.get c.p.P.ins 0 in
  let x = net_operand st (D.Any d) s.nshape in
  let want, got, launches = net_results ~budget:s.nbudget b c s.nshape x in
  let nets = List.filter network_launch launches in
  (* Moments' centers come first, from a map, the scan's only: a cast's kernel
     where the operand is a narrow float's. *)
  let map_launch (t : Plan.traced) =
    directs [ t ] <> []
    ||
    match t.kernel with
    | Loop_map_v4 | Loop_map_v1 | Loop_map_tiled | Loop_map_padded
    | Loop_map_wide ->
        true
    | k -> is_cast (Nx_metal.Kernels.name k)
  in
  let rec leading = function
    | t :: r when map_launch t -> 1 + leading r
    | _ -> 0
  in
  let maps = List.length (List.filter map_launch launches) in
  equal ~msg:"a scan starts with a map where it is Moments" bool (c.r = Moments)
    (leading launches > 0);
  equal ~msg:"map launches after another kernel's" int 0
    (maps - leading launches);
  let v =
    if List.exists (fun (t : Plan.traced) -> t.kernel = Loop_network_v4) nets
    then 4
    else 1
  in
  let w = net_side s.nshape in
  let tile = 256 * v / w in
  let terms = s.nshape.(1) in
  let tiles = terms / tile in
  cover "an odd count of 3 or more whole tiles" (tiles >= 3 && tiles mod 2 = 1);
  cover "a partial last tile" (terms mod tile <> 0);
  cover "past 32 tiles" (tiles > 32);
  cover "a slice's tiles in several launches" (List.length nets > 1);
  cover "one slice a group" (w = 1);
  cover "slices side by side" (w > 1 && w < 32);
  cover "32 slices a group" (w = 32);
  cover "64 to 256 slices a group" (w > 32);
  cover "128 or 256 slices a group" (w >= 128);
  let chains =
    List.length
      (List.filter (fun (t : Plan.traced) -> t.kernel = Loop_carries) launches)
  in
  (* A block's terms, its network launches' NX_LOOP_SEGMENT. *)
  let block =
    match nets with
    | t :: _ -> word t Nx_metal.Kernels.loop_segment
    | [] -> terms
  in
  cover "several blocks a slice" (chains > 0);
  cover "several blocks a slice, slices side by side" (chains > 0 && w > 1);
  cover "a whole block b, 4 dividing b + 1" (terms / block >= 4);
  cover "a slice's blocks' chain in several launches" (chains > 1);
  cover "Moments" (c.r = Moments);
  same_bits want got

(* nx_metal.mli: a 1-D Logsumexp scan of 2^21 terms, past the old network's
   floor, has nx.cpu's bits. *)
let test_long_network () =
  let b = backend () in
  let c = List.hd net_cases in
  let shape = [| 1; 1 lsl 21; 1 |] in
  let x = net_operand (Random.State.make [| 21 |]) (D.Any D.Float32) shape in
  let want, got, _ = net_results b c shape x in
  same_bits want got

(* nx_metal.mli: a Logsumexp or Moments scan of more than one block a slice
   takes 12 bytes of workspace for each block and 24 for each level of its
   blocks' network, one more than the bits of a slice's blocks, for each slice,
   and a Moments scan 4 bytes for each slice more. A block is 4096 terms where a
   group of them fits a launch, as for these programs: 1-D slices of 2^20 and
   2^18 + 1 terms, two slices side by side of 2^16, and three slices of one
   block. *)
let test_network_workspace () =
  let buffer = Lazy.force long_buffer in
  let f32 = D.Any D.Float32 in
  let rec bits n = if n = 0 then 0 else 1 + bits (n lsr 1) in
  List.iter
    (fun (what, p, r, shape) ->
      match long_plan buffer p r ~into:f32 ~shape ~step:false with
      | Error a -> failf "%s: %a" what Nx_array_support.pp_answer a
      | Ok None -> failf "%s: no plan" what
      | Ok (Some (pl : Plan.t)) ->
          let slices = Array.fold_left ( * ) 1 shape / shape.(0) in
          let blocks = (shape.(0) + 4095) / 4096 in
          let network =
            if blocks > 1 then (12 * blocks) + (24 * (bits blocks + 1)) else 0
          in
          let center = match r with Moments -> 4 | _ -> 0 in
          equal
            ~msg:(Format.asprintf "%s, %a" what pp_shape shape)
            int
            (slices * (network + center))
            pl.workspace)
    Nx_kernel.
      [
        ("a float32 Logsumexp", identity f32, Logsumexp, [| 1 lsl 20 |]);
        ("Moments of 4 float32 Pows", pows f32 4, Moments, [| (1 lsl 18) + 1 |]);
        ("float32 Moments", identity f32, Moments, [| 1 lsl 16; 2 |]);
        ("float32 Moments of a block", identity f32, Moments, [| 1000; 3 |]);
      ]

(* Reductions of short rows *)

(* A reduction of class [rc] of [short_cases] along axis 1 of [rshape], its
   operand drawn from [rseed]. *)
type short_rows = { rc : int; rshape : int array; rseed : int }

let short_cases =
  let f32 = D.Any D.Float32 and bf16 = D.Any D.Bfloat16 in
  let i32 = D.Any D.Int32 and i4 = D.Any D.Int4 in
  let case what p r into = { what; p; r; into; carried = false } in
  Nx_kernel.
    [
      case "a float32 sum" (identity f32) (Monoid Sum) f32;
      case "a bfloat16 product" (identity bf16) (Monoid Prod) bf16;
      case "an int32 maximum" (identity i32) (Monoid Max) i32;
      case "an Arg of float32" (identity f32) (Arg Min) f32;
      case "a float32 maximum" (identity f32) (Monoid Max) f32;
      case "an Arg of int4" (identity i4) (Arg Max) i4;
      case "a float32 Logsumexp" (identity f32) Logsumexp f32;
      case "bfloat16 Moments" (identity bf16) Moments bf16;
      case "a sum of a float32 Exp" (exps f32 1) (Monoid Sum) f32;
    ]

let pp_short_rows ppf s =
  Format.fprintf ppf "%s along axis 1 of %a" (List.nth short_cases s.rc).what
    pp_shape s.rshape

(* Rows of 1 to 16 terms, of 1 to 15000 outputs, the terms strided in a third of
   the draws. *)
let short_rows =
  let open Gen in
  let* rc = int_range 0 (List.length short_cases - 1) in
  let* terms = of_list [ 1; 2; 3; 4; 6; 8; 12; 16 ] in
  let* outputs = of_list [ 1; 7; 300; 5000 ] in
  let* after = of_list [ 1; 1; 3 ] in
  let* rseed = int in
  with_pp pp_short_rows
    (constant { rc; rshape = [| outputs; terms; after |]; rseed })

(* Rows of one term and of two, each an eighth of the draws, so that each run
   reaches them. *)
let short_rows_examples =
  [
    { rc = 0; rshape = [| 7; 1; 1 |]; rseed = 1 };
    { rc = 0; rshape = [| 7; 2; 3 |]; rseed = 2 };
  ]

(* nx.cpu's results and nx.metal's of [c]'s reduction along axis 1 of [shape],
   over the host operand [x]. *)
let reduced_results (b : Support.backend) c shape x =
  let dts =
    match c.r with
    | Moments -> [ c.into; c.into ]
    | Arg _ -> [ c.into; D.Any D.Int64 ]
    | Logsumexp | Monoid _ -> [ c.into ]
  in
  let outs = Iarray.of_list [ shape.(0); shape.(2) ] in
  let run (module K : Nx_kernel.S) dev ~on =
    let dsts =
      Array.of_list
        (List.map (fun (D.Any e) -> A.Any (A.create dev e outs)) dts)
    in
    let (A.Any x) = on x in
    let a =
      K.reduce c.p ~shape:(Iarray.of_array shape) ~axes:[| 1 |]
        [| (c.r, 0) |]
        ~dsts:(Iarray.of_array dsts) [| Nx_kernel.Plain x |]
    in
    equal ~msg:(Printf.sprintf "%s: answer" c.what) answer A.Done a;
    dsts
  in
  let want = run (module Nx_cpu) Rig.host ~on:Fun.id in
  let got =
    run (module Nx_metal) b.device ~on:(fun (A.Any a) -> A.Any (Support.on b a))
  in
  (want, got)

(* The products of the host operand [x], of float32 or bfloat16, along axis 1 of
   [shape], in nx.cpu's lane order in float32 and rounded once into [x]'s dtype.
   Where [flush], each combine multiplies as nx_metal.mli says [Mul] does at
   underflow: a subnormal operand reads as a zero of its sign, and a nonzero
   exact result below 2^-126 is a zero of its sign; else as nx.cpu does. Also
   whether a combine read or gave such a value. *)
let lane_products ~flush shape (A.Any x) =
  let narrow =
    match A.dtype x with
    | D.Bfloat16 -> true
    | D.Float32 -> false
    | _ -> invalid_arg "lane_products"
  in
  let bytes = Kernels_cases.bits_of x in
  let width = if narrow then 2 else 4 in
  let value i =
    let b k = bytes.((i * width) + k) in
    let bits =
      if narrow then (b 0 lsl 16) lor (b 1 lsl 24)
      else b 0 lor (b 1 lsl 8) lor (b 2 lsl 16) lor (b 3 lsl 24)
    in
    Int32.float_of_bits (Int32.of_int bits)
  in
  let underflowed = ref false in
  let tiny v = v <> 0. && Float.abs v < Float.ldexp 1. (-126) in
  let flushed v =
    if not (tiny v) then v
    else begin
      underflowed := true;
      if flush then Float.copy_sign 0. v else v
    end
  in
  (* A float32 product is exact in a double, so it rounds once. *)
  let mul a b =
    Int32.float_of_bits (Int32.bits_of_float (flushed (flushed a *. flushed b)))
  in
  let outputs = shape.(0) and terms = shape.(1) and after = shape.(2) in
  let term o a t = value ((((o * terms) + t) * after) + a) in
  let block o a k =
    let lanes = Array.make 16 0. and held = Array.make 16 false in
    for t = 1024 * k to Int.min terms (1024 * (k + 1)) - 1 do
      let l = t mod 16 in
      if held.(l) then lanes.(l) <- mul lanes.(l) (term o a t)
      else begin
        lanes.(l) <- term o a t;
        held.(l) <- true
      end
    done;
    List.iter
      (fun h ->
        for i = 0 to h - 1 do
          if held.(i + h) then lanes.(i) <- mul lanes.(i) lanes.(i + h)
        done)
      [ 8; 4; 2; 1 ];
    lanes.(0)
  in
  let rec tree o a first count =
    if count = 1 then block o a first
    else
      let rec below p = if 2 * p < count then below (2 * p) else p in
      let left = below 1 in
      mul (tree o a first left) (tree o a (first + left) (count - left))
  in
  let blocks = (terms + 1023) / 1024 in
  let y = A.create Rig.host (A.dtype x) [| outputs; after |] in
  let s = Bytes.create (outputs * after * width) in
  for o = 0 to outputs - 1 do
    for a = 0 to after - 1 do
      let i = (o * after) + a in
      let bits = Int32.to_int (Int32.bits_of_float (tree o a 0 blocks)) in
      let bits = bits land 0xffff_ffff in
      if not narrow then Bytes.set_int32_le s (4 * i) (Int32.of_int bits)
      else
        (* Ties to even; a NaN keeps its sign and is quiet. *)
        let top =
          if Float.is_nan (Int32.float_of_bits (Int32.of_int bits)) then
            (bits lsr 16) lor 0x40
          else (bits + 0x7fff + ((bits lsr 16) land 1)) lsr 16
        in
        Bytes.set_uint16_le s (2 * i) (top land 0xffff)
    done
  done;
  Rig.Buffer.blit_from_string (Bytes.to_string s) 0 (A.buffer y) 0
    (Bytes.length s);
  (A.Any y, !underflowed)

(* nx_metal.mli's results of [c]'s reduction along axis 1 of [shape] over [x],
   from nx.cpu's [want]: a product's are the lane order's with the GPU's
   combines, which [want] must equal with nx.cpu's; any other's are [want]. Also
   whether a product's combine underflowed. *)
let metal_want c shape x want =
  match c.r with
  | Monoid Prod ->
      let cpu, underflowed = lane_products ~flush:false shape x in
      same_bits want [| cpu |];
      let gpu, _ = lane_products ~flush:true shape x in
      ([| gpu |], underflowed)
  | Monoid _ | Arg _ | Logsumexp | Moments -> (want, false)

(* nx_metal.mli: a reduction folds its terms in nx.cpu's lane order, an output
   of few terms as one of many, wherever it falls among the outputs. *)
let law_short_rows s =
  let b = backend () in
  let c = List.nth short_cases s.rc in
  let st = Random.State.make [| s.rseed |] in
  let (D.Any d) = Iarray.get c.p.P.ins 0 in
  let x =
    match d with
    | D.Float32 | D.Bfloat16 -> net_operand st (D.Any d) s.rshape
    | _ -> split_operand st (D.Any d) s.rshape
  in
  let want, got = reduced_results b c s.rshape x in
  let terms = s.rshape.(1) in
  cover "one term an output" (terms = 1);
  cover "2 terms an output" (terms = 2);
  cover "3 or 4 terms an output" (terms = 3 || terms = 4);
  cover "5 to 8 terms an output" (terms > 4 && terms <= 8);
  cover "9 to 16 terms an output" (terms > 8);
  cover "more than 4096 outputs" (s.rshape.(0) * s.rshape.(2) > 4096);
  cover "strided terms" (s.rshape.(2) > 1);
  cover "an Arg" (match c.r with Arg _ -> true | _ -> false);
  cover "a Logsumexp or Moments"
    (match c.r with Logsumexp | Moments -> true | Monoid _ | Arg _ -> false);
  let want, _ = metal_want c s.rshape x want in
  same_bits want got

(* A reduction of class [wc] of [window_cases] over windows of [wk] terms on the
   last two axes of a [b; c; h; w] operand [wshape], [wstep] apart, the axes
   padded by [wpad] on each side; its operand's floats all -0 where [wzero],
   else drawn from [wseed] with a NaN of a payload where [wnan]. *)
type short_windows = {
  wc : int;
  wshape : int array;
  wk : int * int;
  wstep : int * int;
  wpad : int;
  wzero : bool;
  wnan : bool;
  wseed : int;
}

(* The classes of short_cases a window takes: a product's lane order is the
   GPU's (metal_want), which only rows of one axis follow. *)
let window_cases =
  List.filter
    (fun c -> match c.r with Nx_kernel.Monoid Prod -> false | _ -> true)
    short_cases

let pp_short_windows ppf s =
  Format.fprintf ppf
    "%s over %dx%d windows %dx%d apart of %a padded by %d%s%s, seed %d"
    (List.nth window_cases s.wc).what (fst s.wk) (snd s.wk) (fst s.wstep)
    (snd s.wstep) pp_shape s.wshape s.wpad
    (if s.wzero then ", all -0" else "")
    (if s.wnan then ", a NaN payload" else "")
    s.wseed

(* The windows' count along an axis of [e] elements padded by [p] on each
   side. *)
let window_count e p k st = ((e + (2 * p) - k) / st) + 1

(* Windows of 1 to 9 terms, a third of them 2x2 of step 2; operands of up to 16
   x 40 x 40 elements a batch element, some padded. *)
let short_windows =
  let open Gen in
  let* wc = int_range 0 (List.length window_cases - 1) in
  let* wk =
    of_list [ (2, 2); (2, 2); (1, 2); (2, 1); (3, 3); (2, 3); (1, 1) ]
  in
  let* square = frequency [ (1, constant true); (2, constant false) ] in
  let wk = if square then (2, 2) else wk in
  let* wstep =
    if square then constant (2, 2) else of_list [ wk; (1, 1); (2, 1); (1, 2) ]
  in
  let* wpad = of_list [ 0; 0; 0; 1 ] in
  let* b = of_list [ 1; 2 ] in
  let* c = of_list [ 1; 3; 16 ] in
  let* h = int_range 1 40 in
  let* w = int_range 1 40 in
  let* wzero = frequency [ (1, constant true); (5, constant false) ] in
  let* wnan = bool in
  let* wseed = int in
  let fits e k = e + (2 * wpad) >= k in
  let h = if fits h (fst wk) then h else fst wk in
  let w = if fits w (snd wk) then w else snd wk in
  with_pp pp_short_windows
    (constant
       { wc; wshape = [| b; c; h; w |]; wk; wstep; wpad; wzero; wnan; wseed })

(* A float32 sum of 2x2 windows over operands all -0, which the draws reach
   about one time in 25. *)
let short_window_examples =
  [
    {
      wc = class_in window_cases "a float32 sum";
      wshape = [| 1; 3; 8; 8 |];
      wk = (2, 2);
      wstep = (2, 2);
      wpad = 0;
      wzero = true;
      wnan = false;
      wseed = 1;
    };
  ]

(* The bits [code], of [x]'s width, at element [i] of [x]'s buffer. *)
let plant (A.Any x) i code =
  let n = D.bytes (A.dtype x) 1 in
  let s = Bytes.create n in
  for k = 0 to n - 1 do
    Bytes.set_uint8 s k ((code lsr (8 * k)) land 0xff)
  done;
  Rig.Buffer.blit_from_string (Bytes.to_string s) 0 (A.buffer x) (n * i) n

(* nx_metal.mli: a reduction of windows of few terms has nx.cpu's bits, by its
   lane order, a NaN's payload included; a window of -0 terms sums as nx.cpu's
   lanes do, each from +0. *)
let law_short_windows s =
  let b = backend () in
  let c = List.nth window_cases s.wc in
  let st = Random.State.make [| s.wseed |] in
  let (D.Any d) = Iarray.get c.p.P.ins 0 in
  let float = D.Any d = D.Any D.Float32 || D.Any d = D.Any D.Bfloat16 in
  let x =
    if float then net_operand st (D.Any d) s.wshape
    else split_operand st (D.Any d) s.wshape
  in
  let n = Array.fold_left ( * ) 1 s.wshape in
  let neg_zero = if D.Any d = D.Any D.Float32 then 0x8000_0000 else 0x8000 in
  if s.wzero && float then
    for i = 0 to n - 1 do
      plant x i neg_zero
    done;
  if s.wnan && float && n > 0 then
    plant x (Random.State.int st n)
      (if D.Any d = D.Any D.Float32 then 0x7fc0_1234 else 0xffc5);
  let kh, kw = s.wk and sh, sw = s.wstep in
  let p = s.wpad in
  let oh = window_count s.wshape.(2) p kh sh
  and ow = window_count s.wshape.(3) p kw sw in
  let loop = [| s.wshape.(0); s.wshape.(1); oh; ow; kh; kw |] in
  let outs = Iarray.of_list [ s.wshape.(0); s.wshape.(1); oh; ow ] in
  let pads = Iarray.of_list [ 0; 0; p; p ] in
  let window axis size step = { M.axis; size; step; dilation = 1 } in
  let pad =
    {
      Nx_kernel.lo = pads;
      hi = pads;
      interior = Iarray.of_list [ 0; 0; 0; 0 ];
      windows = Iarray.of_list [ window 2 kh sh; window 3 kw sw ];
    }
  in
  let dts =
    match c.r with
    | Moments -> [ c.into; c.into ]
    | Arg _ -> [ c.into; D.Any D.Int64 ]
    | Logsumexp | Monoid _ -> [ c.into ]
  in
  let run (module K : Nx_kernel.S) dev ~on =
    let dsts =
      Array.of_list
        (List.map (fun (D.Any e) -> A.Any (A.create dev e outs)) dts)
    in
    let (A.Any x) = on x in
    let fill = D.of_float (A.dtype x) (if s.wzero then -0. else 0.) in
    let a =
      K.reduce c.p ~shape:(Iarray.of_array loop) ~axes:[| 4; 5 |]
        [| (c.r, 0) |]
        ~dsts:(Iarray.of_array dsts)
        [| Nx_kernel.Padded { x; fill; pad } |]
    in
    equal ~msg:(Printf.sprintf "%s: answer" c.what) answer A.Done a;
    dsts
  in
  let want = run (module Nx_cpu) Rig.host ~on:Fun.id in
  let got =
    run (module Nx_metal) b.device ~on:(fun (A.Any a) -> A.Any (Support.on b a))
  in
  cover "a 2x2 window of step 2" (s.wk = (2, 2) && s.wstep = (2, 2) && p = 0);
  cover "a window of -0 terms, summed"
    (s.wzero && match c.r with Monoid Sum -> true | _ -> false);
  cover "a NaN payload" (s.wnan && float);
  cover "a padded window" (p > 0);
  cover "overlapping windows" (sh < kh || sw < kw);
  cover "more than 4096 windows" (Iarray.fold_left ( * ) 1 outs > 4096);
  same_bits want got

(* A reduction of class [pc] of [short_cases] along axis 1 of [pshape], in
   launches of a budget [place] tenths of the way, geometrically, from the least
   under which it is planned to the least under which its groups' runs fit a
   launch whole; its operand drawn from [pseed]. *)
type pieces = { pc : int; pshape : int array; place : int; pseed : int }

let pp_pieces ppf s =
  Format.fprintf ppf "%s along axis 1 of %a, %d tenths, seed %d"
    (List.nth short_cases s.pc).what pp_shape s.pshape s.place s.pseed

(* Outputs of one block of several runs, of up to 16 blocks, and of several
   groups. *)
let pieces =
  let open Gen in
  let* pc = int_range 0 (List.length short_cases - 1) in
  let* terms =
    frequency
      [
        (2, int_range 129 1024);
        (3, int_range 1025 16384);
        (2, int_range 16385 40_000);
      ]
  in
  let* outputs = of_list [ 1; 2; 5 ] in
  let* after = of_list [ 1; 1; 3 ] in
  let* place = int_range 0 9 in
  let* pseed = int in
  with_pp pp_pieces
    (constant { pc; pshape = [| outputs; terms; after |]; place; pseed })

(* The least budget, within 1 %, at which [plan] is [ok]. *)
let least_ok plan ok =
  let fits t = bounded t (fun () -> ok (plan ())) in
  let rec go lo hi =
    if hi /. lo < 1.01 then hi
    else
      let m = sqrt (lo *. hi) in
      if fits m then go lo m else go m hi
  in
  go 1e-8 Plan.budget

(* nx_metal.mli: a group whose terms take longer than a launch allows runs in
   launches of a run of each thread's terms, its threads' states in workspace
   between them, with nx.cpu's bits but where a combine underflows. *)
let law_pieces s =
  let b = backend () in
  let c = List.nth short_cases s.pc in
  let st = Random.State.make [| s.pseed |] in
  let (D.Any d) = Iarray.get c.p.P.ins 0 in
  (* A maximum's terms are specials half the time, so that a launch's piece
     meets a NaN or a zero after another left its state. *)
  let x =
    match d with
    | _ when c.r = Monoid Max -> host_specials st (D.Any d) s.pshape
    | D.Float32 | D.Bfloat16 -> net_operand st (D.Any d) s.pshape
    | _ -> split_operand st (D.Any d) s.pshape
  in
  let plan () =
    let dsts =
      Iarray.init (Nx_metal.Loop.destinations c.r) (fun i ->
          let (D.Any e) =
            if i = 1 && c.r <> Moments then D.Any D.Int64 else c.into
          in
          A.Any (A.create b.device e [| s.pshape.(0); s.pshape.(2) |]))
    in
    let (A.Any x) = x in
    loop_reduce ~holds:(Nx_metal.holds b.device) c.p
      ~shape:(Iarray.of_array s.pshape) ~axes:[| 1 |]
      [| (c.r, 0) |]
      ~dsts
      [| Nx_kernel.Plain (Support.on b x) |]
  in
  let groups =
    match plan () with
    | Ok (Some p) -> reduce_groups p
    | Ok None | Error _ -> failf "%a: not planned" pp_pieces s
  in
  let least = least_ok plan (function Ok (Some _) -> true | _ -> false) in
  let whole =
    least_ok plan (function
      | Ok (Some p) -> reduce_groups p = groups
      | _ -> false)
  in
  let budget = least *. ((whole /. least) ** (Float.of_int s.place /. 10.)) in
  let pieced =
    match bounded budget plan with
    | Ok (Some p) -> reduce_groups p > groups
    | Ok None | Error _ -> false
  in
  let terms = s.pshape.(1) in
  cover "a group's runs in several launches" pieced;
  cover "one block an output" (terms <= 1024);
  cover "2 to 16 blocks an output" (terms > 1024 && terms <= 16384);
  cover "several groups an output" (terms > 16384);
  cover "an Arg" (match c.r with Arg _ -> true | _ -> false);
  cover "a Logsumexp or Moments"
    (match c.r with Logsumexp | Moments -> true | Monoid _ | Arg _ -> false);
  let (want, got), launches =
    bounded budget (fun () ->
        Plan.traced (fun () -> reduced_results b c s.pshape x))
  in
  let by f =
    pieced
    && List.exists
         (fun (t : Plan.traced) -> f (Nx_metal.Kernels.name t.kernel))
         launches
  in
  let row r name = String.starts_with ~prefix:("loop_row_" ^ r) name in
  cover "a group's runs in several launches by loop_reduce_direct"
    (by (( = ) "loop_reduce_direct"));
  cover "a group's runs in several launches by a row's sum" (by (row "sum"));
  cover "a group's runs in several launches by a row's extreme"
    (by (fun n -> List.exists (fun r -> row r n) [ "max"; "min"; "arg" ]));
  let want, underflowed = metal_want c s.pshape x want in
  cover "a product's combine underflows" underflowed;
  same_bits want got

(* The budget of loops *)

(* A loop's launches on the GPU count the groups a core holds by the threads
   Metal lets a group of their pipeline have: a map, a reduction, scans of each
   mode, a network's. *)
let test_resident () =
  let b = backend () in
  let f32 = D.Any D.Float32 in
  let id = P.v ~ins:[| f32 |] [| P.In 0 |] ~outs:[| 0 |] in
  let x shape = A.create b.device D.Float32 (Iarray.of_list shape) in
  let any shape = A.Any (x shape) in
  let load shape = Iarray.of_list [ Nx_kernel.Plain (x shape) ] in
  let calls =
    [
      (fun () -> Nx_metal.map id ~dsts:[| any [ 4096 ] |] (load [ 4096 ]));
      (fun () ->
        Nx_metal.reduce id ~shape:[| 64; 4096 |] ~axes:[| 1 |]
          [| (Monoid Sum, 0) |]
          ~dsts:[| any [ 64 ] |]
          (load [ 64; 4096 ]));
      (fun () ->
        Nx_metal.scan id ~axis:1 (Monoid Sum, 0)
          ~dsts:[| any [ 8; 9000 ] |]
          (load [ 8; 9000 ]));
      (fun () ->
        Nx_metal.scan id ~axis:0 (Monoid Sum, 0)
          ~dsts:[| any [ 300; 2 ] |]
          (load [ 300; 2 ]));
      (fun () ->
        Nx_metal.scan id ~axis:1 (Logsumexp, 0)
          ~dsts:[| any [ 4; 300 ] |]
          (load [ 4; 300 ]));
      (fun () ->
        Nx_metal.scan id ~axis:1 (Moments, 0)
          ~dsts:[| any [ 4; 300 ]; any [ 4; 300 ] |]
          (load [ 4; 300 ]));
    ]
  in
  List.iter
    (fun call ->
      let got, traced = Plan.traced call in
      equal answer A.Done got;
      List.iter
        (fun (t : Plan.traced) ->
          let k = t.kernel in
          let most = Nx_metal.holds b.device k / Nx_metal.Kernels.threads k in
          at_most ~msg:(Nx_metal.Kernels.name k) int ~than:most t.resident)
        traced)
    calls

(* A launch's budget: the library's, a bound below it, or the least, within 0.1
   %, under which the call plans, where a reduction's groups of an output run
   apart. *)
type budget = Library | Bound of float | Least

type looped = {
  family : int;  (** 0 a map, 1 a reduction, 2 a scan. *)
  kind : int;
  ldt : int;
  lshape : int array;
  lbudget : budget;
  lholds : int;  (** The threads Metal lets a group of every kernel have. *)
  lpadded : bool;  (** The first load padded along the last axis. *)
  lheld : bool;  (** A program holding 130 values at once, in parts. *)
  lstep : bool;  (** A scan's results stepping by one along the slice. *)
}

let loop_dtypes =
  D.[ Any Float32; Any Bfloat16; Any Int64; Any Complex64; Any Int4 ]

let pp_looped ppf c =
  let (D.Any dt) = List.nth loop_dtypes c.ldt in
  Format.fprintf ppf "%s kind %d %s [%s]%s, %d threads a group%s%s%s"
    (match c.family with 0 -> "map" | 1 -> "reduce" | _ -> "scan")
    c.kind (D.name dt) (ints c.lshape)
    (match c.lbudget with
    | Library -> ""
    | Bound t -> Printf.sprintf ", %g s" t
    | Least -> ", the least budget")
    c.lholds
    (if c.lpadded then ", padded" else "")
    (if c.lheld then ", 130 values held" else "")
    (if c.lstep then ", stepping by one" else "")

(* Up to 3 axes of up to 2^34 elements in all, broadcast, under the library's
   budget or, up to 2^26 elements, a smaller one, past a group's least work for
   some; and sums of rows of 2 to 64 groups under the least budget they plan
   under. Some pad their first load, some hold more values than a launch holds,
   so that they run in parts, and some scans' results step by one along the
   slice, which a chunk's 16 threads scan, their last axis then of at most 2^16
   elements. A seventh are maps of 130 values held, of up to 2^18 elements under
   the library's budget, which plan, so that every run reaches one in parts. *)
let looped =
  let open Gen in
  let held =
    let+ ldt = int_range 0 (List.length loop_dtypes - 1)
    and+ lshape = array ~size:(int_range 1 3) (int_range 1 64)
    and+ lholds = of_list [ 256; 576; 704; 1024; 3072 ]
    and+ lpadded = bool in
    {
      family = 0;
      kind = 0;
      ldt;
      lshape;
      lbudget = Library;
      lholds;
      lpadded;
      lheld = true;
      lstep = false;
    }
  in
  let apart =
    let+ ldt = int_range 0 (List.length loop_dtypes - 1)
    and+ rows = int_range 1 3
    and+ terms = int_range (1 lsl 15) (1 lsl 20)
    and+ lholds = of_list [ 256; 576; 704; 1024; 3072 ] in
    {
      family = 1;
      kind = 0;
      ldt;
      lshape = [| rows; terms |];
      lbudget = Least;
      lholds;
      lpadded = false;
      lheld = false;
      lstep = false;
    }
  in
  let any =
    let extent =
      frequency
        [
          (6, int_range 1 300);
          (3, int_range 1 (1 lsl 14));
          (1, int_range 1 (1 lsl 24));
        ]
    in
    let+ family = int_range 0 2
    and+ kind = int_range 0 3
    and+ ldt = int_range 0 (List.length loop_dtypes - 1)
    and+ lshape =
      map
        (fun s ->
          (* In floats: three extents of up to 2^24 multiply past max_int. *)
          let n = Array.fold_left (fun n e -> n *. Float.of_int e) 1. s in
          if n <= 0x1p34 then s else [| 1 lsl 34 |])
        (array ~size:(int_range 1 3) extent)
    and+ lbudget =
      frequency
        [
          (2, constant Library);
          (1, map (fun t -> Bound t) (float_range 2e-5 2e-4));
        ]
    and+ lholds = of_list [ 256; 576; 704; 1024; 3072 ]
    and+ lpadded = frequency [ (3, constant false); (1, constant true) ]
    and+ lheld = frequency [ (3, constant false); (1, constant true) ]
    and+ lstep = bool in
    (* A budget of 2e-5 s splits 2^34 elements of a costly kind into some 10^6
       launches, each holding its parameters. *)
    let n = Array.fold_left ( * ) 1 lshape in
    let lbudget = if n > 1 lsl 26 then Library else lbudget in
    let lstep = lstep && family = 2 in
    let lshape =
      if not lstep then lshape
      else
        let r = Array.length lshape in
        Array.mapi
          (fun i e -> if i = r - 1 then Int.min e (1 lsl 16) else e)
          lshape
    in
    { family; kind; ldt; lshape; lbudget; lholds; lpadded; lheld; lstep }
  in
  with_pp pp_looped (frequency [ (5, any); (1, apart); (1, held) ])

(* Cases each run reaches, which draws reach in 5 to 14 % of the cases: a
   complex64 Exp and Pow of 2^34 elements, past the call's budget; a scan
   stepping by one, its chunks of 16 threads; a sum of rows of 2^18 terms under
   the least budget, merged apart. *)
let looped_examples =
  let c ?(lbudget = Library) ?(lstep = false) family kind ldt lshape =
    {
      family;
      kind;
      ldt;
      lshape;
      lbudget;
      lholds = 1024;
      lpadded = false;
      lheld = false;
      lstep;
    }
  in
  [
    c 0 3 3 [| 1 lsl 17; 1 lsl 17 |];
    c ~lstep:true 2 0 0 [| 3; 5000 |];
    c ~lbudget:Least 1 0 0 [| 2; 1 lsl 18 |];
  ]

(* The plan of [c]'s call over broadcast host arrays: a map of a kind of one
   operand and another of two, or of 130 values held, a reduction of the last
   axis, or a scan along it, into a destination whose last axis steps by one
   where [c.lstep]. *)
let loop_plan c =
  let (D.Any dt) = List.nth loop_dtypes c.ldt in
  let shape = Iarray.of_array c.lshape in
  let r = Array.length c.lshape in
  let arr dt s = broadcast dt s (Rig.Buffer.create Rig.host 16) in
  let x = arr dt shape in
  let ins = Iarray.of_list [ D.Any dt; D.Any dt ] in
  let float = D.is D.Float dt || D.is D.Complex dt in
  let u : P.unary =
    if float then if c.kind mod 2 = 0 then Tan else Exp else Neg
  in
  let bin : P.binary = if float && c.kind = 3 then Pow else Add in
  let p =
    if c.lheld then
      let n = 130 in
      let nodes =
        Array.concat
          [
            [| P.In 0; P.In 1; P.Op2 (Binary Add, 0, 1) |];
            Array.init (n - 1) (fun j -> P.Op2 (Binary Add, j + 2, 0));
            Array.init (n - 1) (fun j ->
                P.Op2 (Binary Add, (if j = 0 then n + 1 else n + 1 + j), n - j));
          ]
      in
      P.v ~ins (Iarray.of_array nodes) ~outs:[| Array.length nodes - 1 |]
    else
      P.v ~ins
        [|
          P.In 0; P.In 1; P.Op1 (Unary u, D.Any dt, 0); P.Op2 (Binary bin, 2, 1);
        |]
        ~outs:[| 3 |]
  in
  let first =
    if not (c.lpadded && c.lshape.(r - 1) >= 2) then Nx_kernel.Plain x
    else
      let inner =
        Array.mapi (fun i e -> if i = r - 1 then e - 1 else e) c.lshape
      in
      let zeros = Iarray.init r (fun _ -> 0) in
      let lo = Iarray.init r (fun i -> if i = r - 1 then 1 else 0) in
      Nx_kernel.Padded
        {
          x = arr dt (Iarray.of_array inner);
          fill = D.of_float dt 0.;
          pad =
            { lo; hi = zeros; interior = zeros; windows = Iarray.of_list [] };
        }
  in
  let loads = [| first; Nx_kernel.Plain (arr dt shape) |] in
  (* A destination whose last axis steps by one, the others broadcast. *)
  let stepped (D.Any d) =
    let last = c.lshape.(r - 1) in
    let strides = Iarray.init r (fun i -> if i = r - 1 then 1 else 0) in
    let bytes = Int.max 16 (((last * D.bits d) + 7) / 8) in
    A.Any (A.v d (L.v ~strides shape) (Rig.Buffer.create Rig.host bytes))
  in
  let scan_dst d s =
    if c.lstep then stepped d
    else
      let (D.Any d) = d in
      A.Any (arr d s)
  in
  let reduction : Nx_kernel.reduction =
    match c.kind with
    | 0 -> Monoid Sum
    | 1 -> Arg Max
    | 2 when D.equal_witness dt D.Float32 <> None -> Logsumexp
    | 3 when D.equal_witness dt D.Float32 <> None -> Moments
    | _ -> Monoid Max
  in
  let holds _ = c.lholds in
  let plan () =
    match c.family with
    | 0 ->
        loop_map ~holds p
          ~dsts:[| A.Any (arr dt shape) |]
          (Iarray.of_array loads)
    | 1 ->
        let want = Iarray.of_array (Array.sub c.lshape 0 (r - 1)) in
        let dsts =
          Array.init (Nx_metal.Loop.destinations reduction) (fun i ->
              if i = 0 then A.Any (arr dt want) else A.Any (arr D.Int64 want))
        in
        let dsts =
          match reduction with
          | Moments -> [| A.Any (arr dt want); A.Any (arr dt want) |]
          | _ -> dsts
        in
        loop_reduce ~holds p ~shape
          ~axes:[| r - 1 |]
          [| (reduction, 0) |]
          ~dsts:(Iarray.of_array dsts) (Iarray.of_array loads)
    | _ ->
        let dsts =
          Array.init (Nx_metal.Loop.destinations reduction) (fun i ->
              if i = 0 then scan_dst (D.Any dt) shape
              else scan_dst (D.Any D.Int64) shape)
        in
        let dsts =
          match reduction with
          | Moments ->
              [| scan_dst (D.Any dt) shape; scan_dst (D.Any dt) shape |]
          | _ -> dsts
        in
        loop_scan ~holds p ~axis:(r - 1) (reduction, 0)
          ~dsts:(Iarray.of_array dsts) (Iarray.of_array loads)
  in
  let planned t = match bounded t plan with Ok _ -> true | Error _ -> false in
  let rec least lo hi =
    if hi /. lo < 1.001 then hi
    else
      let m = sqrt (lo *. hi) in
      if planned m then least lo m else least m hi
  in
  match c.lbudget with
  | Library -> (plan (), Plan.budget)
  | Bound t -> (bounded t plan, t)
  | Least ->
      let t =
        if planned Plan.budget then least 1e-7 Plan.budget else Plan.budget
      in
      (bounded t plan, t)

(* plan.mli: each launch of a loop keeps within the time budget and each part of
   several launches within a part's, the loop split into launches, or declined
   where a group's least work does not fit; a core holds no more groups than the
   threads Metal lets a group have. *)
let law_loop_budget c =
  let plan, allowed = loop_plan c in
  let n = Array.fold_left ( * ) 1 c.lshape in
  cover "past 2^32 elements" (n > 1 lsl 32);
  cover "declined, a group past its budget"
    (match plan with Error A.Declined -> true | _ -> false);
  match plan with
  | Error A.Declined -> ()
  | Error a -> failf "answered %a" Nx_array_support.pp_answer a
  | Ok None -> ()
  | Ok (Some (p : Plan.t)) ->
      cover "a map" (c.family = 0);
      cover "a reduction" (c.family = 1);
      cover "a scan" (c.family = 2);
      cover "several launches" (Array.length p.launches > 1);
      cover "merged apart"
        (Array.exists
           (fun (l : Plan.launch) -> l.kernel = Nx_metal.Kernels.Loop_merge)
           p.launches);
      cover "a padded load"
        (c.lpadded && c.lshape.(Array.length c.lshape - 1) >= 2);
      cover "a map in parts, through temporaries"
        (c.lheld && c.family = 0
        && (p.workspace > 0
           || Array.exists
                (function Plan.Fresh _ -> true | _ -> false)
                p.slots));
      cover "a scan of 16 threads a chunk"
        (c.lstep && c.lshape.(Array.length c.lshape - 1) >= 16);
      let time (l : Plan.launch) =
        Plan.seconds
          ~threads:(Nx_metal.Kernels.threads l.kernel)
          ~groups:l.groups ~resident:l.resident l.cost
      in
      Array.iter
        (fun (l : Plan.launch) ->
          at_most
            ~msg:(Nx_metal.Kernels.name l.kernel)
            float_exact ~than:allowed (time l);
          at_most ~msg:"groups a core holds" int ~than:(c.lholds / 256)
            l.resident)
        p.launches;
      Array.iter
        (fun (q : Plan.part) ->
          if Array.length q.run > 1 then
            at_most ~msg:"a part's launches together" float_exact
              ~than:Plan.part_budget
              (Array.fold_left (fun t l -> t +. time l) 0. q.run))
        p.parts

(* Contraction plans *)

module C = Nx_metal.Contract

let stage_name = function
  | C.F32 -> "float32"
  | C.F16 -> "float16"
  | C.Bf16 -> "bfloat16"
  | C.I8 -> "bytes of integers"

let tile_name = function
  | C.T64 -> "64x64"
  | C.T32 -> "32x32"
  | C.T16 -> "16x64"
  | C.G32 -> "grouped 32x64"
  | C.G16 -> "grouped 16x64"

let pp_extents ppf (e : C.extents) =
  Format.fprintf ppf "%d x %d x %d x %d %s, b %s%s" e.batch e.m e.n e.k
    (stage_name e.stage)
    (if e.nk then "n x k" else "k x n")
    (if e.complex then ", complex64's real form" else "")

type contracted = { e : C.extents; cut : int }

let pp_contracted ppf c =
  Format.fprintf ppf "%a, budget a %dth" pp_extents c.e c.cut

(* Extents from a row to thousands, k from none to past 2^16 terms; the calls'
   multiply-adds stay under 2^37, so that a budget a few hundred times below a
   call's time cuts it into at most thousands of launches. *)
let contracted =
  let open Gen in
  let extent =
    frequency
      [ (3, int_range 0 20); (3, int_range 17 300); (2, int_range 300 4096) ]
  in
  let depth =
    frequency
      [
        (1, constant 0);
        (3, int_range 1 300);
        (3, int_range 300 8192);
        (1, int_range 8192 70000);
      ]
  in
  let batch =
    frequency [ (4, constant 1); (2, int_range 2 8); (1, int_range 9 128) ]
  in
  with_pp pp_contracted
    (let+ b = batch
     and+ m = extent
     and+ n = extent
     and+ k = depth
     and+ stage = of_list [ C.F32; C.F16; C.Bf16; C.I8 ]
     and+ nk = bool
     and+ complex = frequency [ (4, constant false); (1, constant true) ]
     and+ cut = frequency [ (1, constant 1); (3, int_range 2 300) ] in
     let k = Int.min k ((1 lsl 37) / (b * Int.max 1 m * Int.max 1 n)) in
     (* complex64's real form: twice the call's columns and terms. *)
     let stage, n, k =
       if complex then (C.F32, n / 2 * 2, k / 2 * 2) else (stage, n, k)
     in
     { e = { C.batch = b; m; n; k; stage; nk; complex; source = 0 }; cut })

let cdiv a b = (a + b - 1) / b

(* design: the tiles' threadgroup memory and G, 32 times the groups a core holds
   by it, with b k x n. *)
let spread_of = function
  | C.F32, C.T64 | (C.F16 | C.I8), C.T64 -> 192
  | C.Bf16, C.T64 -> 384
  | C.F32, C.T32 -> 416
  | (C.F16 | C.I8), C.T32 -> 384
  | C.Bf16, C.T32 -> 736
  | C.F32, (C.T16 | C.G16) -> 160
  | (C.F16 | C.Bf16 | C.I8), (C.T16 | C.G16) -> 352
  | (C.F32 | C.F16 | C.I8), C.G32 -> 288
  | C.Bf16, C.G32 -> 544

(* The plan's fields the rule of contract.mli gives, but its launches and the
   groups a core holds. *)
let shape (p : C.t) =
  Printf.sprintf
    "%s%s tiles of %d x %d outputs, %d x %d tiles, swizzle %d, %d a batch \
     element, %d ranges of %d spans"
    (if p.exchanged then "exchanged, " else "")
    (tile_name p.tile) p.rows_out p.cols_out p.tiles_m p.tiles_n p.swizzle
    p.padded p.ranges p.spans

let ruled (e : C.extents) =
  let exchanged = (not e.complex) && e.n <= 16 && e.m > 16 in
  let rows_out, cols_out = if exchanged then (e.n, e.m) else (e.m, e.n) in
  let tile =
    if rows_out <= 16 then C.T16
    else if e.batch * cdiv rows_out 64 * cdiv cols_out 64 >= 96 then C.T64
    else C.T32
  in
  let bm, bn = (C.rows tile, C.cols tile) in
  let tiles_m = cdiv rows_out bm and tiles_n = cdiv cols_out bn in
  let t = e.batch * tiles_m * tiles_n in
  let g = spread_of (e.stage, tile) in
  let spans = cdiv e.k 32 in
  let ranges =
    if t = 0 || 2 * t >= g then 1
    else Int.max 1 (Int.min 64 (Int.min (spans / 8) (cdiv g t)))
  in
  let swizzle = if tiles_m >= 8 && tiles_n >= 8 then 2 else 0 in
  let padded = cdiv tiles_m (1 lsl swizzle) * (tiles_n lsl swizzle) in
  Printf.sprintf
    "%s%s tiles of %d x %d outputs, %d x %d tiles, swizzle %d, %d a batch \
     element, %d ranges of %d spans"
    (if exchanged then "exchanged, " else "")
    (tile_name tile) rows_out cols_out tiles_m tiles_n swizzle padded ranges
    spans

let planned e =
  match C.plan e with
  | Some p -> p
  | None -> failf "%a has no plan under the library's budget" pp_extents e

(* contract.mli: the tile, the ranges and the call's order follow the rule from
   the logical extents and the operands' dtype, whatever b's layout. *)
let law_plan c =
  let p = planned c.e in
  cover "exchanged" p.exchanged;
  cover "complex64, few columns" (c.e.complex && c.e.n <= 16 && c.e.m > 16);
  cover "64x64" (p.tile = C.T64);
  cover "32x32" (p.tile = C.T32);
  cover "16x64" (p.tile = C.T16);
  cover "several ranges" (p.ranges > 1);
  cover "swizzled" (p.swizzle > 0);
  cover "no term" (c.e.k = 0);
  cover "no output" (p.rows_out * p.cols_out = 0);
  cover "a batch" (c.e.batch > 1);
  equal string (ruled c.e) (shape p);
  let other = planned { c.e with nk = not c.e.nk } in
  equal ~msg:"another layout" string (shape p) (shape other)

let estimate e p l =
  Plan.dispatch
  +. C.estimate e p ~bandwidth:Plan.bandwidth ~cores:Plan.cores
       ~op_rate:Plan.op_rate l

(* The estimate of [tiles] tiles over a chunk of [length] terms, whole ranges at
   [0], resuming at [chunk > 0]. *)
let probe e p ~tiles ~chunk ~length =
  estimate e p { C.first = 0; tiles; chunk; length; seconds = 0. }

(* A margin for the estimates' rounding, which the test computes in another
   order than the plan. *)
let tight = 1e-9

(* The windows of a plan's launches: each one's first tile, tiles and chunks, in
   order. *)
let windows (p : C.t) =
  Array.fold_right
    (fun (l : C.launch) acc ->
      match acc with
      | (first, tiles, chunks) :: rest when first = l.first && tiles = l.tiles
        ->
          (first, tiles, (l.chunk, l.length) :: chunks) :: rest
      | _ -> (l.first, l.tiles, [ (l.chunk, l.length) ]) :: acc)
    p.launches []

(* contract.mli: under any budget, the plan cuts the call's tiles, in its order,
   into windows of whole units, each covering its tiles once over each range,
   whole or in chunks in increasing k, each launch within the budget and as long
   as it allows; the tiles and ranges are those of the library's budget. *)
let law_windows c =
  let lib = planned c.e in
  let allowed =
    Float.min Plan.budget (Float.max 1e-8 (lib.seconds /. Float.of_int c.cut))
  in
  let got = bounded allowed (fun () -> C.plan c.e) in
  let total = c.e.batch * lib.padded in
  let block = lib.tiles_n lsl lib.swizzle in
  let g = spread_of (c.e.stage, lib.tile) in
  let rec gcd a b = if b = 0 then a else gcd b (a mod b) in
  let units = [ g / gcd g block * block; block; 1 ] in
  let whole = Int.min c.e.k (32 * cdiv lib.spans (Int.max 1 lib.ranges)) in
  let at = probe c.e lib in
  let broad_by m u =
    at ~tiles:(Int.min total u) ~chunk:0 ~length:0 <= allowed *. (1. +. m)
  in
  let deep_by m u =
    at ~tiles:(Int.min total u) ~chunk:1 ~length:32 <= allowed *. (1. +. m)
  in
  let broad = broad_by tight in
  let unit_by m = List.find_opt (fun u -> broad_by m u || deep_by m u) units in
  (* A unit whose estimate lies within the rounding margin of the budget fits by
     one rounding and not by another, and the plan, which rounds its own, may
     take it or the next: with a budget a 9th of a call of 9 whole units, one
     unit's estimate came 1 ulp over the budget. The plan's unit is the larger
     where its windows are of it. *)
  let unit_in (p : C.t) =
    match (unit_by tight, unit_by (-.tight)) with
    | Some u, Some v when u <> v ->
        let size = match windows p with (_, size, _) :: _ -> size | [] -> 0 in
        if size mod u = 0 || size = Int.min total u then Some u else Some v
    | loose, _ -> loose
  in
  cover "no launch" (total = 0 || c.e.k = 0);
  match got with
  | None ->
      cover "declined" true;
      equal ~msg:"a unit that fits" (option int) None (unit_by (-.tight))
  | Some p ->
      equal ~msg:"the tiles and ranges" string (shape lib) (shape p);
      Array.iter
        (fun (l : C.launch) ->
          at_most ~msg:"a launch" float_exact ~than:allowed l.seconds;
          equal ~msg:"its estimate"
            (float_rel ~rel:tight ~abs:0.)
            (estimate c.e p l) l.seconds)
        p.launches;
      equal ~msg:"the call's time" float_exact
        (Array.fold_left (fun s (l : C.launch) -> s +. l.seconds) 0. p.launches)
        p.seconds;
      let ws = windows p in
      let ends =
        List.fold_left
          (fun at (first, tiles, _) ->
            equal ~msg:"a window's first tile" int at first;
            at_least ~msg:"a window's tiles" int ~than:1 tiles;
            first + tiles)
          0 ws
      in
      if total = 0 || c.e.k = 0 then
        equal ~msg:"launches" int 0 (List.length ws)
      else begin
        equal ~msg:"the tiles covered" int total ends;
        let unit = require_some ~msg:"a unit that fits" (unit_in p) in
        let _, size, chunks = List.hd ws in
        let length = snd (List.hd chunks) in
        let count = if length = 0 then 1 else cdiv whole length in
        let want = List.init count (fun i -> (i, length)) in
        List.iter
          (fun (_, _, cs) ->
            equal ~msg:"a window's chunks" (list (pair int int)) want cs)
          ws;
        let n = List.length ws in
        List.iteri
          (fun i (_, tiles, _) ->
            if i < n - 1 then equal ~msg:"a window's tiles" int size tiles)
          ws;
        cover "several windows" (n > 1);
        cover "chunks" (length > 0);
        cover "units of a block or a tile" (unit < List.hd units);
        if broad unit then begin
          equal ~msg:"whole ranges" int 0 length;
          if n > 1 then begin
            equal ~msg:"a multiple of the unit" int 0 (size mod unit);
            greater ~msg:"one unit more" float_exact
              ~than:(allowed *. (1. -. tight))
              (at ~tiles:(size + unit) ~chunk:0 ~length:0)
          end
        end
        else begin
          equal ~msg:"windows of a unit" int (Int.min total unit) size;
          equal ~msg:"chunks of spans" int 0 (length mod 32);
          if length + 32 <= 32 * cdiv whole 32 then
            greater ~msg:"a span more" float_exact
              ~than:(allowed *. (1. -. tight))
              (at ~tiles:size ~chunk:1 ~length:(length + 32))
        end
      end

type grouped_call = { ge : C.extents; groups : int }

let pp_grouped_call ppf c =
  Format.fprintf ppf "%a in %d groups" pp_extents c.ge c.groups

let grouped_call =
  let open Gen in
  with_pp pp_grouped_call
    (let+ c = contracted
     and+ groups =
       frequency
         [ (1, constant 0); (2, int_range 1 64); (1, int_range 65 20000) ]
     in
     let e = c.e in
     let stage = if e.stage = C.I8 then C.F32 else e.stage in
     { ge = { e with stage; complex = false }; groups })

(* contract.mli: a Rows call's plan takes 32 x 64 tiles where m >= 16 g, else 16
   x 64, the bound's tile rows of all batch elements' rows in [batch g + 1]
   buckets, no swizzle, and S of 3.2 from those tiles, whatever b's layout;
   every launch within the budget. *)
let law_grouped_plan c =
  let e = c.ge and g = c.groups in
  match C.grouped e ~groups:g with
  | None -> fail "no plan"
  | Some p ->
      let tile = if e.m >= 16 * g then C.G32 else C.G16 in
      cover "32x64" (tile = C.G32);
      cover "16x64" (tile = C.G16);
      cover "several ranges" (p.ranges > 1);
      cover "past a chunk of buckets"
        ((e.batch * g) + 1 > Nx_metal.Kernels.buckets_chunk);
      let rows = e.batch * e.m in
      let tiles_m =
        if rows = 0 then 0
        else cdiv rows (C.rows tile) + Int.min ((e.batch * g) + 1) rows - 1
      in
      let tiles_n = cdiv e.n 64 in
      let t = tiles_m * tiles_n in
      let spread = spread_of (e.stage, tile) in
      let spans = cdiv e.k 32 in
      let ranges =
        if t = 0 || 2 * t >= spread then 1
        else Int.max 1 (Int.min 64 (Int.min (spans / 8) (cdiv spread t)))
      in
      equal ~msg:"the tile" string (tile_name tile) (tile_name p.tile);
      equal ~msg:"tile rows" int tiles_m p.tiles_m;
      equal ~msg:"tile columns" int tiles_n p.tiles_n;
      equal ~msg:"swizzle" int 0 p.swizzle;
      equal ~msg:"padded" int t p.padded;
      equal ~msg:"ranges" int ranges p.ranges;
      equal ~msg:"exchanged" bool false p.exchanged;
      Array.iter
        (fun (l : C.launch) ->
          at_most ~msg:"a launch" float_exact ~than:Plan.budget l.seconds)
        p.launches;
      let other = C.grouped { e with nk = not e.nk } ~groups:g in
      equal ~msg:"another layout" (option string)
        (Some (shape p))
        (Option.map shape other)

let call ?(cut = 1) ?(stage = C.F32) ?(nk = false) ?(complex = false) batch m n
    k =
  { e = { C.batch; m; n; k; stage; nk; complex; source = 0 }; cut }

(* Calls that reach each case of the rule: no output, no term, the exchange,
   each tile, ranges up to 64 and a swizzled batch. *)
let plan_examples =
  [
    call 1 0 64 64;
    call 1 64 64 0;
    call 1 4096 8 4096;
    call 1 4096 4096 4096;
    call ~stage:C.Bf16 1 768 768 768;
    call ~stage:C.F16 1 512 512 512;
    call ~complex:true 1 4096 16 4096;
    call 1 16 64 65536;
    call 3 512 512 64;
  ]

(* Calls that a budget cuts each way: into windows of whole ranges, into chunks
   in units of a block, of a tile, and below one tile's 32 terms; and a budget
   at a unit's edge. *)
let window_examples =
  [
    call 1 4096 4096 4096;
    call ~cut:300 1 4096 4096 4096;
    call ~cut:300 1 4096 4096 1024;
    call ~cut:300 1 16 16 64;
    call ~cut:20 ~stage:C.F16 ~nk:true 1 8 4096 4096;
    call 1 0 64 64;
    (* A budget a 9th of a call of 9 swizzled blocks of tiles: one block's
       estimate lies 1 ulp over it. *)
    call ~cut:9 ~stage:C.F16 1 2049 2561 16;
  ]

(* design: the launches of the calls its table states, by their tiles, on the
   M1. *)
let launch_rows =
  [
    ("4096³ float32", call 1 4096 4096 4096, [ 768; 768; 768; 768; 768; 256 ]);
    ( "4096³ float16",
      call ~stage:C.F16 1 4096 4096 4096,
      [ 768; 768; 768; 768; 768; 256 ] );
    ("2048³ float32", call 1 2048 2048 2048, [ 1024 ]);
    ("64 x 64 x 65536 float32, 64 ranges", call 1 64 64 65536, [ 4 ]);
    ("1 x 4096² float32", call 1 1 4096 4096, [ 64 ]);
  ]

let test_launches (_, c, want) =
  let p = planned c.e in
  let tiles =
    Array.to_list (Array.map (fun (l : C.launch) -> l.tiles) p.launches)
  in
  equal ~msg:"the launches' tiles" (list int) want tiles

(* Contraction bits *)

(* nx_metal.mli's fused multiply-add at float32: its operands read flushed, [a b
   + c] rounded once, and a zero of its sign where the exact result is nonzero
   and below 2^-126. The product is exact in a double; the sum is the double [s]
   and its error [t] (s + t exact), rounded to float32 by {!round32}. Products
   are held apart from the sums that follow them, which ocamlopt would fuse. A
   reference sums hundreds of millions of these, so they take no C call. *)
let[@inline] flush32 x =
  if Float.abs x < 0x1p-126 then Float.copy_sign 0. x else x

let[@inline] to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* 2^⌊log2 |x|⌋ of a finite nonzero [x] below 2^970: Rump's ufp, from x (2^52 +
   1) less its rounding times 1 - 2^-53. *)
let[@inline] ufp x =
  let q = Sys.opaque_identity (0x1.0000000000001p52 *. x) in
  Float.abs (q -. Sys.opaque_identity (0x1.fffffffffffffp-1 *. q))

(* [s + t] to float32, ties to even, past its largest value infinity, where [s]
   is the double nearest it, at least 2^-126 in magnitude. With u = ufp s,
   adding and taking away k = 1.5 2^29 u rounds s to a multiple of u 2^-23,
   float32's spacing about s, ties to even: s + k lies in k's binade, whose
   spacing is that. That is s + t's rounding but where s is a tie, half a
   spacing from the result, and t <> 0: then s + t lies past the tie on t's
   side, and s moved a quarter spacing that way rounds as it does. No other tie
   lies between s and s + t, as |t| is at most half of s's spacing as a
   double. *)
let[@inline] round32 s t =
  let u = ufp s in
  let k = Sys.opaque_identity (0x1.8p29 *. u) in
  let y = Sys.opaque_identity (s +. k) -. k in
  let y =
    if t = 0. || Float.abs (s -. y) <> 0x1p-24 *. u then y
    else Sys.opaque_identity (s +. Float.copy_sign (0x1p-25 *. u) t +. k) -. k
  in
  if Float.abs y >= 0x1p128 then Float.copy_sign infinity y else y

let[@inline] fma32 a b c =
  let a = flush32 a and b = flush32 b and c = flush32 c in
  let d = Sys.opaque_identity (a *. b) in
  let s = d +. c in
  if not (Float.is_finite d && Float.is_finite c) then to_f32 s
  else
    let bb = s -. d in
    let t = d -. (s -. bb) +. (c -. bb) in
    let below =
      Float.abs s < 0x1p-126
      || Float.abs s = 0x1p-126
         && t <> 0.
         && Float.sign_bit t <> Float.sign_bit s
    in
    if s = 0. then s else if below then Float.copy_sign 0. s else round32 s t

let[@inline] add32 x y = fma32 x 1. y

(* nx_metal.mli's association: range s of r = ⌈k / 32⌉ spans holds the terms
   from 32 ⌊s r / S⌋ to 32 ⌊(s + 1) r / S⌋, at most k, each summed from +0 by
   fused multiply-adds in increasing k; the ranges added in order from the
   first, then init. An output of no product is init, or +0. *)
let associated ~ranges ~k ~init term =
  let r = (k + 31) / 32 in
  let sum = ref 0. in
  for s = 0 to ranges - 1 do
    let k0 = 32 * (s * r / ranges)
    and k1 = Int.min k (32 * ((s + 1) * r / ranges)) in
    let part = ref 0. in
    for kk = k0 to k1 - 1 do
      let a, b = term kk in
      part := fma32 a b !part
    done;
    sum := if s = 0 then !part else add32 !sum !part
  done;
  match init with
  | None -> if k = 0 then 0. else !sum
  | Some i -> if k = 0 then i else add32 !sum i

(* {!associated} of the terms [(a.(a0 + kk sa), b.(b0 + kk sb))], as a loop over
   the arrays: the same sums, without a term's closure or its pair. *)
let associated_at ~ranges ~k ~init a a0 sa b b0 sb =
  let r = (k + 31) / 32 in
  let sum = ref 0. in
  for s = 0 to ranges - 1 do
    let k0 = 32 * (s * r / ranges)
    and k1 = Int.min k (32 * ((s + 1) * r / ranges)) in
    let part = ref 0. in
    for kk = k0 to k1 - 1 do
      part :=
        fma32
          (Array.unsafe_get a (a0 + (kk * sa)))
          (Array.unsafe_get b (b0 + (kk * sb)))
          !part
    done;
    sum := if s = 0 then !part else add32 !sum !part
  done;
  match init with
  | None -> if k = 0 then 0. else !sum
  | Some i -> if k = 0 then i else add32 !sum i

(* {!associated_at} of each of [outputs] sums, output [o]'s terms from [at o] in
   [a] and [b] and its init [init o]: four at a time, which the processor
   overlaps, where one sum alone waits on each term's rounding. *)
let associated_all ~ranges ~k ~init ~outputs ~at a sa b sb =
  let r = (k + 31) / 32 in
  let close o sum =
    match init o with
    | None -> if k = 0 then 0. else sum
    | Some i -> if k = 0 then i else add32 sum i
  in
  let sums = Array.make outputs 0. in
  let four = outputs / 4 * 4 in
  for q = 0 to (four / 4) - 1 do
    let o = 4 * q in
    let a0, b0 = at o and a1, b1 = at (o + 1) in
    let a2, b2 = at (o + 2) and a3, b3 = at (o + 3) in
    let s0 = ref 0. and s1 = ref 0. and s2 = ref 0. and s3 = ref 0. in
    for s = 0 to ranges - 1 do
      let k0 = 32 * (s * r / ranges)
      and k1 = Int.min k (32 * ((s + 1) * r / ranges)) in
      let p0 = ref 0. and p1 = ref 0. and p2 = ref 0. and p3 = ref 0. in
      for kk = k0 to k1 - 1 do
        let ka = kk * sa and kb = kk * sb in
        p0 :=
          fma32
            (Array.unsafe_get a (a0 + ka))
            (Array.unsafe_get b (b0 + kb))
            !p0;
        p1 :=
          fma32
            (Array.unsafe_get a (a1 + ka))
            (Array.unsafe_get b (b1 + kb))
            !p1;
        p2 :=
          fma32
            (Array.unsafe_get a (a2 + ka))
            (Array.unsafe_get b (b2 + kb))
            !p2;
        p3 :=
          fma32
            (Array.unsafe_get a (a3 + ka))
            (Array.unsafe_get b (b3 + kb))
            !p3
      done;
      if s = 0 then begin
        s0 := !p0;
        s1 := !p1;
        s2 := !p2;
        s3 := !p3
      end
      else begin
        s0 := add32 !s0 !p0;
        s1 := add32 !s1 !p1;
        s2 := add32 !s2 !p2;
        s3 := add32 !s3 !p3
      end
    done;
    sums.(o) <- close o !s0;
    sums.(o + 1) <- close (o + 1) !s1;
    sums.(o + 2) <- close (o + 2) !s2;
    sums.(o + 3) <- close (o + 3) !s3
  done;
  for o = four to outputs - 1 do
    let a0, b0 = at o in
    sums.(o) <- associated_at ~ranges ~k ~init:(init o) a a0 sa b b0 sb
  done;
  sums

(* The reference of an unsegmented call of [bt] x [m] x [n] x [k] whose terms
   are [av z i kk] and [bv z kk j], held once as arrays in C order of [bt; m; k]
   and [bt; k; n]: those arrays, every output's sum, output [o]'s sum by
   {!associated_at}, or with term [kk] made +0 where [drop], and its terms. *)
let reference ~ranges ~bt ~m ~n ~k ~init av bv =
  let a =
    Array.init (bt * m * k) (fun t -> av (t / (m * k)) (t / k mod m) (t mod k))
  in
  let b =
    Array.init (bt * k * n) (fun t -> bv (t / (k * n)) (t / n mod k) (t mod n))
  in
  let at o =
    let z = o / (m * n) and i = o / n mod m and j = o mod n in
    (((z * m) + i) * k, (z * k * n) + j)
  in
  let init o = Option.map (fun v -> v.(o)) init in
  let sum ?drop o =
    let a0, b0 = at o in
    match drop with
    | None -> associated_at ~ranges ~k ~init:(init o) a a0 1 b b0 n
    | Some kk ->
        associated ~ranges ~k ~init:(init o) (fun t ->
            if t = kk then (0., 0.) else (a.(a0 + t), b.(b0 + (t * n))))
  in
  let terms o =
    let a0, b0 = at o in
    Array.init k (fun t -> (a.(a0 + t), b.(b0 + (t * n))))
  in
  let sums =
    associated_all ~ranges ~k ~init ~outputs:(bt * m * n) ~at a 1 b n
  in
  (a, b, sums, sum, terms)

type stage_dt = Stage : (float, 's) D.t -> stage_dt

let stage_dts =
  [
    ("float32", Stage D.Float32);
    ("float16", Stage D.Float16);
    ("bfloat16", Stage D.Bfloat16);
  ]

(* Each staged dtype's specials: zeros, infinities, a NaN, its least subnormal
   and the normal numbers about 2^-126. *)
let specials = function
  | "float16" ->
      [|
        0.; -0.; infinity; neg_infinity; nan; 0x1p-24; -0x1p-24; 0x1p-14; 65504.;
      |]
  | "bfloat16" ->
      [|
        0.;
        -0.;
        infinity;
        neg_infinity;
        nan;
        0x1p-133;
        0x1p-126;
        -0x1p-126;
        0x1.02p-126;
      |]
  | _ ->
      [|
        0.;
        -0.;
        infinity;
        neg_infinity;
        nan;
        0x1p-149;
        -0x1p-149;
        0x1p-126;
        0x1.000002p-126;
        0x1.fffffcp-127;
      |]

type bits = {
  stage : int;
  bnk : bool;
  at : bool;
  bbatch : int;
  bm : int;
  bn : int;
  bk : int;
  low : int;
  special : int;
  binit : bool;
  bdst : int;
  bseed : int;
  few : bool;
}

let pp_bits ppf c =
  let name i = fst (List.nth stage_dts i) in
  Format.fprintf ppf
    "%d x %d x %d x %d %s%s, b %s%s, values from 2^%d, specials 1/%d, %sdst \
     %s, seed %d"
    c.bbatch c.bm c.bn c.bk (name c.stage)
    (if c.few then ", 64 x 64 from one tile" else "")
    (if c.bnk then "n x k" else "k x n")
    (if c.at then ", a strided" else "")
    c.low c.special
    (if c.binit then "init, " else "")
    (name c.bdst) c.bseed

(* Extents about each tile's edges and k about its depths and spans; the values'
   exponents from [low] to [low + 16], products about 2^-126 from [low] =
   -71. *)
let bits_case =
  let open Gen in
  let rows =
    frequency
      [ (2, int_range 1 16); (2, int_range 17 80); (1, int_range 81 200) ]
  in
  let depth =
    frequency
      [
        (1, constant 0);
        (3, int_range 1 80);
        (2, int_range 81 700);
        (1, of_list [ 33; 40; 2048 ]);
      ]
  in
  with_pp pp_bits
    (let+ stage = int_range 0 2
     and+ bnk = bool
     and+ at = frequency [ (3, constant false); (1, constant true) ]
     and+ bbatch = frequency [ (3, constant 1); (1, int_range 2 3) ]
     and+ bm = rows
     and+ bn = rows
     and+ bk = depth
     and+ low = frequency [ (3, constant (-8)); (1, constant (-71)) ]
     and+ special = of_list [ 4; 16; 1000000 ]
     and+ binit = bool
     and+ bdst = int_range 0 2
     and+ bseed = nat in
     {
       stage;
       bnk;
       at;
       bbatch;
       bm;
       bn;
       bk;
       low;
       special;
       binit;
       bdst;
       bseed;
       few = false;
     })

let drawn_values st ~name ~low ~special n =
  let sp = specials name in
  Array.init n (fun _ ->
      if Random.State.int st special = 0 then
        sp.(Random.State.int st (Array.length sp))
      else
        let e = low + Random.State.int st 16 in
        let v = Float.ldexp (1. +. Random.State.float st 1.) e in
        if Random.State.bool st then v else -.v)

(* An array of [dt] on the host holding [vs] as [dt] rounds them, and the values
   it holds. *)
let stored (type s) (dt : (float, s) D.t) shape vs =
  let x = A.of_array dt (Iarray.of_list shape) vs in
  (x, A.to_array x)

let library_plan c =
  let stage = List.nth [ C.F32; C.F16; C.Bf16 ] c.stage in
  planned
    {
      C.batch = c.bbatch;
      m = c.bm;
      n = c.bn;
      k = c.bk;
      stage;
      nk = c.bnk;
      complex = false;
      source = 0;
    }

(* [f ()], where [few] with 64 x 64 tiles from a call's first tile on
   (Contract.bound_big): a call of one batch element and a few tiles then runs
   the 64 x 64 tile's kernels, which the library gives only to calls of 96
   tiles, 393,216 outputs or more. *)
let tiles_64 few f =
  if not few then f ()
  else begin
    C.bound_big 1;
    Fun.protect ~finally:(fun () -> C.bound_big C.big) f
  end

(* Computes [c] on the GPU and checks every output's bits against the stated
   association: the plan of the library's budget, whose tiles and ranges no
   budget changes, and the launches the call planned. *)
let contracted_bits c =
  let bk = backend () in
  let name, Stage dt = List.nth stage_dts c.stage in
  let st = Random.State.make [| c.bseed |] in
  let draw n = drawn_values st ~name ~low:c.low ~special:c.special n in
  let bt = c.bbatch and m = c.bm and n = c.bn and k = c.bk in
  (* Each operand, its values, and where a row's or a column's terms start in
     them and their step. *)
  let a, av, a_at =
    if c.at then
      let x, v = stored dt [ bt; k; m ] (draw (bt * k * m)) in
      let x = Option.get (A.move (M.Permute (Iarray.of_list [ 0; 2; 1 ])) x) in
      (x, v, fun z i -> ((z * k * m) + i, m))
    else
      let x, v = stored dt [ bt; m; k ] (draw (bt * m * k)) in
      (x, v, fun z i -> (((z * m) + i) * k, 1))
  in
  let b, bv, b_at =
    if c.bnk then
      let x, v = stored dt [ bt; n; k ] (draw (bt * n * k)) in
      (x, v, fun z j -> (((z * n) + j) * k, 1))
    else
      let x, v = stored dt [ bt; k; n ] (draw (bt * k * n)) in
      (x, v, fun z j -> ((z * k * n) + j, n))
  in
  let outputs = bt * m * n in
  let dname, Stage ddt = List.nth stage_dts c.bdst in
  let init =
    if not c.binit then None
    else
      let vs =
        drawn_values st ~name:dname ~low:c.low ~special:c.special outputs
      in
      Some (stored ddt [ bt; m; n ] vs)
  in
  let dst = A.create (S.device ()) ddt (Iarray.of_list [ bt; m; n ]) in
  let p = library_plan c in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (2, if c.bnk then 2 else 1) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
      { Nx_kernel.array = A.Any (Support.on bk a); scale = None }
      { Nx_kernel.array = A.Any (Support.on bk b); scale = None }
  in
  equal answer A.Done got;
  (* Each operand's terms are one step apart in every output. *)
  let sa = snd (a_at 0 0) and sb = snd (b_at 0 0) in
  let want =
    associated_all ~ranges:p.ranges ~k ~outputs
      ~init:(fun o -> Option.map (fun (_, v) -> v.(o)) init)
      ~at:(fun o ->
        let z = o / (m * n) and i = o / n mod m and j = o mod n in
        (fst (a_at z i), fst (b_at z j)))
      av sa bv sb
  in
  let want = A.to_array (A.of_array ddt (Iarray.of_list [ bt; m; n ]) want) in
  equal (array float_exact) want (A.to_array (Support.host dst));
  launches

(* Whether [k] is a float tile, an integer one, or a grouped one. *)
let is_tile k =
  List.exists
    (fun prefix -> String.starts_with ~prefix (Nx_metal.Kernels.name k))
    [ "contract_f32_"; "contract_f16_"; "contract_bf16_" ]

let int_tile k =
  String.starts_with ~prefix:"contract_i8_" (Nx_metal.Kernels.name k)

let grouped_tile k =
  String.starts_with ~prefix:"contract_rows_" (Nx_metal.Kernels.name k)

let tile_launch (t : Plan.traced) = is_tile t.kernel

(* The tile functions a contraction takes: 36, three for each stage and tile
   with b read k x n (one for calls whose k fills the tile's steps and whose
   columns fill its tiles, one for other calls of at least a tile's columns, one
   for calls of fewer), one with b read n x k. The float32 tiles that read a
   routine's options, plain and negated, are the matrix routines' alone
   (test_metal_product), and the banded twins a banded call's
   (law_mixed_pieces). *)
let banded_twin k = String.ends_with ~suffix:"_band" (Nx_metal.Kernels.name k)

let tile_kernels =
  let routine k =
    List.exists
      (fun prefix -> String.starts_with ~prefix (Nx_metal.Kernels.name k))
      [ "contract_f32_rt_"; "contract_f32_neg_" ]
  in
  List.filter
    (fun k -> is_tile k && (not (routine k)) && not (banded_twin k))
    (Array.to_list Nx_metal.Kernels.all)

(* nx_metal.mli: every output of a float contraction is the stated association
   of its products at float32, each fused multiply-add flushed as the subnormal
   rule states, then init, rounded once into dst: bit for bit, whatever b's
   layout or a's strides, at every staged dtype, tile, edge and count of
   ranges. *)
let law_bits c =
  tiles_64 c.few @@ fun () ->
  let p = library_plan c in
  let name = fst (List.nth stage_dts c.stage) in
  cover "float16" (name = "float16");
  cover "bfloat16" (name = "bfloat16");
  cover "exchanged" p.exchanged;
  cover "16x64" (c.bk > 0 && p.tile = C.T16);
  cover "32x32" (c.bk > 0 && p.tile = C.T32);
  cover "64x64" (c.bk > 0 && p.tile = C.T64);
  cover "a copied" c.at;
  cover "b n x k" c.bnk;
  cover "two ranges" (p.ranges = 2);
  cover "more ranges" (p.ranges > 2);
  cover "a partial step" (c.bk mod 16 <> 0);
  cover "an init" c.binit;
  cover "no product" (c.bk = 0);
  let launches = contracted_bits c in
  List.iter
    (fun k ->
      cover (Nx_metal.Kernels.name k)
        (List.exists (fun (t : Plan.traced) -> t.kernel = k) launches))
    tile_kernels;
  (* nx_metal.mli: sums take workspace and a last pass into dst where there is
     no term, more than one range, an init, or a dst of neither float32 nor the
     operands' dtype. *)
  let passes =
    c.bk = 0 || p.ranges > 1 || c.binit || (c.bdst <> 0 && c.bdst <> c.stage)
  in
  cover "a last pass" passes;
  cover "into dst" (not passes);
  equal ~msg:"a last pass" bool passes
    (List.exists
       (fun (t : Plan.traced) -> t.kernel = Nx_metal.Kernels.Contract_finish)
       launches)

let bits_of ?(few = false) ?(stage = 0) ?(bnk = false) ?(at = false)
    ?(binit = false) ?(bdst = 0) bbatch bm bn bk =
  {
    stage;
    bnk;
    at;
    bbatch;
    bm;
    bn;
    bk;
    low = -8;
    special = 16;
    binit;
    bdst;
    bseed = 7;
    few;
  }

(* k x n extents (batch, m, n) about each tile's columns: the 16 x 64 tile's at
   n short of a thread's run of 16 columns and at it, short of a tile, at it and
   past it, and two tiles whose last starts at an odd column; the 32 x 32 tile's
   at n under a tile and about it, its last tile at an odd column; the 64 x 64
   tile's about a tile and its last at an odd column, from one tile on. *)
let column_edges =
  [
    (false, 5, 15);
    (false, 5, 16);
    (false, 5, 63);
    (false, 5, 64);
    (false, 5, 65);
    (false, 5, 129);
    (false, 40, 17);
    (false, 40, 31);
    (false, 40, 32);
    (false, 40, 33);
    (false, 40, 47);
    (true, 64, 63);
    (true, 64, 65);
    (true, 64, 97);
  ]

(* Calls that reach each tile function, at each stage, layout of b and tile, for
   k that fills the tile's steps (32) and k that does not (40), the 64 x 64
   tile's at one batch element and at two; the exchange, ranges of two and more,
   and no product, at each staged dtype; and a bfloat16 call whose spread gives
   it 8 ranges where float32's would give 5. *)
let bits_examples =
  List.concat_map
    (fun stage ->
      List.concat_map
        (fun (bnk, bk) ->
          [
            bits_of ~few:true ~stage ~bnk 1 256 256 bk;
            bits_of ~few:true ~stage ~bnk 2 128 128 bk;
            bits_of ~stage ~bnk 1 100 128 bk;
            bits_of ~stage ~bnk 1 16 64 bk;
          ])
        [ (false, 32); (false, 40); (true, 32); (true, 40) ])
    [ 0; 1; 2 ]
  @ [
      bits_of ~stage:2 2 200 200 2048;
      bits_of ~few:true ~stage:1 ~bnk:true 1 250 260 72;
      bits_of ~few:true ~stage:2 ~binit:true ~bdst:2 1 256 200 33;
      bits_of ~bnk:true 1 200 9 100;
      bits_of ~stage:1 ~bdst:1 1 16 64 600;
      bits_of ~stage:2 ~at:true 1 8 70 2048;
      bits_of ~binit:true ~bdst:1 2 30 40 0;
    ]
  @ List.concat_map
      (fun stage ->
        List.map (fun (few, m, n) -> bits_of ~few ~stage 1 m n 40) column_edges)
      [ 0; 1; 2 ]

(* Complex contraction bits *)

(* How a complex call's operands are viewed: fresh; b broadcast along the batch;
   a and b at an offset; b's rows stepped; or the terms over two contracted
   axes, b's stepped along the first, so that they do not merge and b is copied
   first. *)
type cview = Fresh | Broadcast | Offset | Stepped | Split

type cbits = {
  cbatch : int;
  cm : int;
  cn : int;
  ck : int;
  cat : bool;
  cbt : bool;
  cinit : bool;
  cview : cview;
  clow : int;
  cspecial : int;
  cseed : int;
}

let pp_cbits ppf c =
  let view =
    match c.cview with
    | Fresh -> ""
    | Broadcast -> ", b broadcast"
    | Offset -> ", at an offset"
    | Stepped -> ", b stepped"
    | Split -> ", terms over two axes, b's not merging"
  in
  Format.fprintf ppf
    "%d x %d x %d x %d complex64%s%s%s%s, values from 2^%d, specials 1/%d, \
     seed %d"
    c.cbatch c.cm c.cn c.ck
    (if c.cat then ", a strided" else "")
    (if c.cbt then ", b n x k" else "")
    view
    (if c.cinit then ", init" else "")
    c.clow c.cspecial c.cseed

(* Rows and columns about the tiles' edges, a real form's columns twice the
   call's, and k about the spans and depths. *)
let cbits_case =
  let open Gen in
  let extent =
    frequency
      [ (2, int_range 1 16); (2, int_range 17 80); (1, int_range 81 160) ]
  in
  let depth =
    frequency
      [
        (1, constant 0);
        (3, int_range 1 60);
        (2, int_range 61 400);
        (1, of_list [ 8; 16; 1024 ]);
      ]
  in
  with_pp pp_cbits
    (let+ cbatch = frequency [ (3, constant 1); (1, int_range 2 3) ]
     and+ cm = extent
     and+ cn = extent
     and+ ck = depth
     and+ cat = frequency [ (3, constant false); (1, constant true) ]
     and+ cbt = bool
     and+ cinit = bool
     and+ cview =
       frequency
         [
           (3, constant Fresh);
           (1, constant Broadcast);
           (1, constant Offset);
           (1, constant Stepped);
           (1, constant Split);
         ]
     and+ clow = frequency [ (3, constant (-8)); (1, constant (-71)) ]
     and+ cspecial = of_list [ 4; 16; 1000000 ]
     and+ cseed = nat in
     let cbt = cbt && cview <> Split in
     { cbatch; cm; cn; ck; cat; cbt; cinit; cview; clow; cspecial; cseed })

(* [c]'s terms: under [Split], two axes of 2 and [ck / 2]. *)
let cterms c = if c.cview = Split then 2 * (c.ck / 2) else c.ck

(* The plan of [c]'s real form: m rows, 2n columns, 2k terms. *)
let complex_plan ~nk c =
  planned
    {
      C.batch = c.cbatch;
      m = c.cm;
      n = 2 * c.cn;
      k = 2 * cterms c;
      stage = C.F32;
      nk;
      complex = true;
      source = 0;
    }

(* A host complex64 array of [shape] holding [vs]'s pairs as float32 rounds
   them, and its values. *)
let stored_complex shape vs =
  let n = Array.length vs / 2 in
  let x =
    A.of_array D.Complex64 (Iarray.of_list shape)
      (Array.init n (fun i ->
           { Complex.re = vs.(2 * i); im = vs.((2 * i) + 1) }))
  in
  (x, A.to_array x)

(* A complex64 view of [shape] = [bt; r; c] from [draw], as [view] makes it, its
   elements moved by [on], and its element at (z, p, q). [Broadcast] repeats one
   batch element; [Offset] starts 3 elements into its buffer; [Stepped] keeps
   every other row of what [on] moved, since a copy to another device packs the
   rows it skips ({!Nx_array.to_device}). *)
let viewed_complex ~on view draw shape =
  let bt, r, c =
    match shape with
    | [ bt; r; c ] -> (bt, r, c)
    | _ -> invalid_arg "viewed_complex"
  in
  let moved mv x = Option.get (A.move mv x) in
  let range start count step = { M.start; count; step } in
  match view with
  | Fresh ->
      let x, v = stored_complex shape (draw (bt * r * c)) in
      (on x, fun z p q -> v.((((z * r) + p) * c) + q))
  | Broadcast ->
      let x, v = stored_complex [ 1; r; c ] (draw (r * c)) in
      ( moved (M.Broadcast (Iarray.of_list shape)) (on x),
        fun _ p q -> v.((p * c) + q) )
  | Offset ->
      let skip = 3 and n = bt * r * c in
      let x, v = stored_complex [ skip + n ] (draw (skip + n)) in
      let x = moved (M.Slice [| range skip n 1 |]) (on x) in
      ( moved (M.Reshape (Iarray.of_list shape)) x,
        fun z p q -> v.(skip + (((z * r) + p) * c) + q) )
  | Stepped ->
      let x, v = stored_complex [ bt; 2 * r; c ] (draw (bt * 2 * r * c)) in
      let x =
        moved (M.Slice [| range 0 bt 1; range 0 r 2; range 0 c 1 |]) (on x)
      in
      (x, fun z p q -> v.((((z * 2 * r) + (2 * p)) * c) + q))
  | Split -> invalid_arg "viewed_complex: Split"

(* [Split]'s b of [bt] batch elements, [k] terms and [n] columns, as [bt; 2; k /
   2; n], its first terms' axis every other of four of what [on] moved, and its
   element at (z, p, q), p numbered in C order of the two axes. *)
let split_complex ~on draw bt k n =
  let h = k / 2 in
  let x, v = stored_complex [ bt; 4; h; n ] (draw (bt * 4 * h * n)) in
  let x = on x in
  let range start count step = { M.start; count; step } in
  let x =
    Option.get
      (A.move
         (M.Slice [| range 0 bt 1; range 0 2 2; range 0 h 1; range 0 n 1 |])
         x)
  in
  (x, fun z p q -> v.((((((z * 4) + (2 * (p / h))) * h) + (p mod h)) * n) + q))

(* nx_metal.mli: a complex64 contraction is the float32 contraction of its real
   form, term 2p of each part its product p's first, term 2p + 1 its second:
   ar·br then ai·(-bi) in the real part, ar·bi then ai·br in the imaginary part,
   each part in the stated association of 2k terms, then init's part, rounded
   once into dst. Never exchanged, whatever its columns. The plan is that of b's
   layout as the tiles read it, which the traced tile names. *)
let law_complex_bits c =
  let bk = backend () in
  let st = Random.State.make [| c.cseed |] in
  let draw n =
    drawn_values st ~name:"float32" ~low:c.clow ~special:c.cspecial (2 * n)
  in
  let bt = c.cbatch and m = c.cm and n = c.cn and k = cterms c in
  let split = c.cview = Split in
  let on x = Support.on bk x in
  let moved mv x = Option.get (A.move mv x) in
  let permuted x = moved (M.Permute (Iarray.of_list [ 0; 2; 1 ])) x in
  (* Under [Split], a's terms as two axes of 2 and k / 2. *)
  let a_terms x =
    if not split then x
    else if c.cat then
      let x = moved (M.Reshape (Iarray.of_list [ bt; 2; k / 2; m ])) x in
      moved (M.Permute (Iarray.of_list [ 0; 3; 1; 2 ])) x
    else moved (M.Reshape (Iarray.of_list [ bt; m; 2; k / 2 ])) x
  in
  let a_view = if c.cview = Offset then Offset else Fresh in
  let a, av =
    if c.cat then
      let x, v = viewed_complex ~on a_view draw [ bt; k; m ] in
      ((if split then a_terms x else permuted x), fun z i kk -> v z kk i)
    else
      let x, v = viewed_complex ~on a_view draw [ bt; m; k ] in
      (a_terms x, fun z i kk -> v z i kk)
  in
  let b, bv =
    if split then split_complex ~on draw bt k n
    else if c.cbt then
      let x, v = viewed_complex ~on c.cview draw [ bt; n; k ] in
      (permuted x, fun z kk j -> v z j kk)
    else
      let x, v = viewed_complex ~on c.cview draw [ bt; k; n ] in
      (x, fun z kk j -> v z kk j)
  in
  let outputs = bt * m * n in
  let init =
    if c.cinit then Some (stored_complex [ bt; m; n ] (draw outputs)) else None
  in
  let dst = A.create (S.device ()) D.Complex64 (Iarray.of_list [ bt; m; n ]) in
  let contracting = if split then [ (2, 1); (3, 2) ] else [ (2, 1) ] in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list contracting)
      ~segments:None ~acc:(D.Any D.Complex64) ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
      { Nx_kernel.array = A.Any (Support.on bk a); scale = None }
      { Nx_kernel.array = A.Any (Support.on bk b); scale = None }
  in
  equal answer A.Done got;
  let tile =
    List.find_map
      (fun (t : Plan.traced) ->
        let s = Nx_metal.Kernels.name t.kernel in
        if String.starts_with ~prefix:"contract_c64_" s then Some s else None)
      launches
  in
  let nk =
    Option.fold ~none:false ~some:(String.ends_with ~suffix:"_nk") tile
  in
  let p = complex_plan ~nk c in
  cover "16x64" (c.ck > 0 && p.tile = C.T16);
  cover "32x32" (c.ck > 0 && p.tile = C.T32);
  cover "64x64" (c.ck > 0 && p.tile = C.T64);
  cover "64x64, b n x k" (tile = Some "contract_c64_64_nk");
  cover "16x64, b n x k, its last rows clamped"
    (tile = Some "contract_c64_16_nk" && 2 * n mod 64 <> 0);
  cover "16x64, b k x n, narrow" (tile = Some "contract_c64_16_kn_n");
  cover "64x64, b k x n, narrow" (tile = Some "contract_c64_64_kn_n");
  cover "few columns, many rows" (c.cn <= 16 && c.cm > 16);
  cover "several ranges" (p.ranges > 1);
  cover "an init" c.cinit;
  cover "no product" (c.ck = 0);
  cover "a strided" c.cat;
  cover "b n x k" c.cbt;
  cover "b broadcast along the batch"
    (c.cview = Broadcast && c.cbatch > 1 && c.ck > 0);
  cover "operands at an offset" (c.cview = Offset && c.ck > 0);
  cover "b's rows stepped" (c.cview = Stepped && c.ck > 0);
  cover "b's terms not merging, copied"
    (split && k > 0 && (not c.cat)
    && List.exists (fun (t : Plan.traced) -> apart_copy t.kernel) launches);
  equal ~msg:"exchanged" bool false p.exchanged;
  (* The real form's 2k terms: a row's ar ai pairs; b's column as br -bi pairs
     for the real part, bi br for the imaginary one. *)
  let real_a =
    Array.init
      (bt * m * 2 * k)
      (fun q ->
        let t = q mod (2 * k) and zi = q / (2 * k) in
        let x = av (zi / m) (zi mod m) (t / 2) in
        if t mod 2 = 0 then x.Complex.re else x.im)
  in
  let real_b im =
    Array.init
      (bt * n * 2 * k)
      (fun q ->
        let t = q mod (2 * k) and zj = q / (2 * k) in
        let y = bv (zj / n) (t / 2) (zj mod n) in
        match (t mod 2 = 0, im) with
        | true, false -> y.Complex.re
        | false, false -> -.y.im
        | true, true -> y.im
        | false, true -> y.re)
  in
  let part im =
    associated_all ~ranges:p.ranges ~k:(2 * k) ~outputs
      ~init:(fun o ->
        Option.map
          (fun (_, v) -> if im then v.(o).Complex.im else v.(o).Complex.re)
          init)
      ~at:(fun o ->
        let z = o / (m * n) and i = o / n mod m and j = o mod n in
        (((z * m) + i) * 2 * k, ((z * n) + j) * 2 * k))
      real_a 1 (real_b im) 1
  in
  let got = A.to_array (Support.host dst) in
  equal ~msg:"real parts" (array float_exact) (part false)
    (Array.map (fun (x : Complex.t) -> x.re) got);
  equal ~msg:"imaginary parts" (array float_exact) (part true)
    (Array.map (fun (x : Complex.t) -> x.im) got)

let cbits_of ?(cat = false) ?(cbt = false) ?(cinit = false) ?(cview = Fresh)
    cbatch cm cn ck =
  {
    cbatch;
    cm;
    cn;
    ck;
    cat;
    cbt;
    cinit;
    cview;
    clow = -8;
    cspecial = 16;
    cseed = 11;
  }

(* Calls that reach each tile, aligned and not, b n x k at 64 x 64 and at 16 x
   64 past its last row, b k x n narrow at 16 x 64 and 64 x 64, few columns
   beside many rows, several ranges, no product, each kind of operand, and each
   view: b broadcast along a batch, operands at an offset, b's rows stepped, and
   terms over two axes, b's not merging. *)
let cbits_examples =
  [
    cbits_of 6 256 128 32;
    cbits_of 6 256 128 20;
    cbits_of ~cbt:true 6 256 128 32;
    cbits_of ~cbt:true 1 16 40 16;
    cbits_of 1 12 20 24;
    cbits_of 24 256 20 16;
    cbits_of ~cbt:true 1 100 64 16;
    cbits_of ~cat:true 1 100 70 17;
    cbits_of 1 16 32 8;
    cbits_of ~cinit:true 1 9 50 33;
    cbits_of 1 200 8 40;
    cbits_of 1 16 32 2048;
    cbits_of ~cinit:true 2 10 12 0;
    cbits_of ~cview:Broadcast 3 40 24 50;
    cbits_of ~cview:Broadcast ~cbt:true 3 20 70 33;
    cbits_of ~cview:Offset 2 30 20 40;
    cbits_of ~cview:Offset ~cat:true ~cbt:true 1 17 9 25;
    cbits_of ~cview:Stepped 2 25 30 40;
    cbits_of ~cview:Stepped ~cbt:true 1 40 16 64;
    cbits_of ~cview:Split 2 30 20 40;
    cbits_of ~cview:Split ~cat:true 1 17 70 66;
  ]

(* Integer contraction bits *)

let int_accs =
  D.
    [
      Any Int8;
      Any Uint8;
      Any Int16;
      Any Uint16;
      Any Int32;
      Any Uint32;
      Any Int64;
      Any Uint64;
      Any Int4;
      Any Uint4;
    ]

(* Operands the half units take: integers of 8 bits or fewer and booleans. *)
let narrow_ints =
  D.[ Any Int8; Any Uint8; Any Int4; Any Uint4; Any Bool; Any Bit ]

(* Operands of the other integer contractions: wider integers and floats. *)
let wide_operands =
  D.
    [
      Any Int16;
      Any Uint16;
      Any Int32;
      Any Uint32;
      Any Int64;
      Any Uint64;
      Any Float32;
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
    ]

let int_dsts = int_accs @ D.[ Any Float32; Any Bfloat16; Any Bool ]

type ibits = {
  iacc : D.any;
  ia : D.any;
  ib : D.any;
  idst : D.any;
  ibatch : int;
  im : int;
  in_ : int;
  ik : int;
  inn : bool;
  iinit : bool;
  ifull : bool;
  iseed : int;
  ifew : bool;
}

let pp_ibits ppf c =
  let name (D.Any d) = D.name d in
  Format.fprintf ppf
    "%d x %d x %d x %d, %s x %s%s%s%s into %s, dst %s%s, seed %d" c.ibatch c.im
    c.in_ c.ik (name c.ia) (name c.ib)
    (if c.inn then " (b n x k)" else "")
    (if c.ifull then " at their maxima" else "")
    (if c.ifew then ", 64 x 64 from one tile" else "")
    (name c.iacc) (name c.idst)
    (if c.iinit then ", init" else "")
    c.iseed

let ibits_case =
  let open Gen in
  let extent =
    frequency
      [ (2, int_range 1 16); (2, int_range 17 80); (1, int_range 81 140) ]
  in
  let depth =
    frequency
      [
        (1, constant 0);
        (3, int_range 1 40);
        (3, of_list [ 255; 256; 257; 513 ]);
        (2, int_range 300 700);
      ]
  in
  let pick l = of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) l in
  let operands =
    frequency
      [
        (2, constant narrow_ints);
        (2, constant wide_operands);
        (1, constant (narrow_ints @ wide_operands));
      ]
  in
  with_pp pp_ibits
    (let* ops = operands in
     let+ iacc = pick int_accs
     and+ ia = pick ops
     and+ ib = pick ops
     and+ same = frequency [ (2, constant true); (1, constant false) ]
     and+ dst = pick int_dsts
     and+ ibatch = frequency [ (3, constant 1); (1, int_range 2 3) ]
     and+ im = extent
     and+ in_ = extent
     and+ ik = depth
     and+ inn = bool
     and+ iinit = bool
     and+ iseed = nat in
     let idst = if same then iacc else dst in
     {
       iacc;
       ia;
       ib;
       idst;
       ibatch;
       im;
       in_;
       ik;
       inn;
       iinit;
       ifull = false;
       iseed;
       ifew = false;
     })

(* The width and signedness of an integer dtype. *)
let int_form (D.Any d) =
  (D.bits d, match D.kind d with D.Signed -> true | _ -> false)

(* [v] modulo 2^w, sign-extended where signed: the integer cast of nx_kinds.h
   into a dtype of [w] bits. *)
let wrap (w, signed) v =
  if w >= 64 then v
  else
    let s = 64 - w in
    if signed then Int64.shift_right (Int64.shift_left v s) s
    else Int64.shift_right_logical (Int64.shift_left v s) s

(* nx_kinds.h's cast of a float into an integer of [w] bits: toward zero,
   saturated, NaN to 0, as the integer's bits sign-extended. *)
let float_into (w, signed) v =
  if Float.is_nan v then 0L
  else
    let t = Float.trunc v in
    let lo = if signed then -.Float.ldexp 1. (w - 1) else 0. in
    let top = if signed then Float.ldexp 1. (w - 1) else Float.ldexp 1. w in
    if t <= lo then Int64.of_float lo
    else if t >= top then
      if signed then Int64.pred (Int64.shift_left 1L (w - 1))
      else if w = 64 then -1L
      else Int64.pred (Int64.shift_left 1L w)
    else if t >= 0x1p63 then
      Int64.add (Int64.of_float (t -. 0x1p63)) Int64.min_int
    else Int64.of_float t

let pick_from st a = a.(Random.State.int st (Array.length a))

(* Values about every integer width's extremes, or small. *)
let int_draw st =
  if Random.State.int st 3 > 0 then
    pick_from st
      [|
        Int64.min_int;
        Int64.max_int;
        0x7FL;
        0x80L;
        0xFFL;
        0x7FFFL;
        0x8000L;
        0xFFFFL;
        0x7FFFFFFFL;
        0x80000000L;
        0xFFFFFFFFL;
        -1L;
        7L;
        8L;
        -8L;
        15L;
        -128L;
        127L;
      |]
  else Int64.of_int (Random.State.int st 16 - 8)

(* Floats about every integer width's bounds, fractions, and specials. *)
let float_draw st =
  if Random.State.int st 3 > 0 then
    pick_from st
      [|
        0.;
        -0.;
        0.5;
        -0.5;
        -0.9;
        2.5;
        -3.75;
        127.5;
        128.;
        255.;
        256.;
        -129.;
        1e10;
        -1e10;
        0x1p31;
        0x1p32;
        0x1p63;
        -0x1p63;
        0x1p64;
        infinity;
        neg_infinity;
        nan;
      |]
  else Float.of_int (Random.State.int st 16 - 8)

(* A boolean byte: 0, 1, or any of 0..255. *)
let bool_byte st =
  match Random.State.int st 4 with
  | 0 -> 0
  | 1 -> 1
  | _ -> Random.State.int st 256

(* The greatest integer of [w] bits, signed or not. *)
let int_max (w, signed) =
  if signed then Int64.pred (Int64.shift_left 1L (w - 1))
  else if w = 64 then -1L
  else Int64.pred (Int64.shift_left 1L w)

(* A host array of [dt] and [shape] from [st], and its values as [acc] holds
   them: an integer's cast, a float's by {!float_into}, a boolean's 1 where its
   byte is not zero (dtype.mli). Booleans are bytes over 0..255 read through
   [Nx_array.bitcast]. Where [full], every element of an integer or boolean [dt]
   is its greatest. *)
let int_operand ?(full = false) st (D.Any dt as d) shape ~acc =
  let n = List.fold_left ( * ) 1 shape in
  let sh = Iarray.of_list shape in
  let form = int_form acc in
  match dt with
  | D.Bool ->
      let byte () = if full then 255 else bool_byte st in
      let bytes = Array.init n (fun _ -> byte ()) in
      let x = Option.get (A.bitcast D.Bool (A.of_array D.Uint8 sh bytes)) in
      (A.Any x, Array.map (fun b -> if b = 0 then 0L else 1L) bytes)
  | _ -> (
      match D.kind dt with
      | D.Float ->
          if full then invalid_arg "int_operand: a float at its maximum";
          let made =
            A.of_array D.Float64 sh (Array.init n (fun _ -> float_draw st))
          in
          let x = Contract_cases.cast_to d (A.Any made) in
          let back = Contract_cases.cast_to (D.Any D.Float64) x in
          let vs = A.to_array (A.expect D.Float64 back) in
          (x, Array.map (float_into form) vs)
      | _ ->
          let draw () = if full then int_max (int_form d) else int_draw st in
          let made = A.of_array D.Int64 sh (Array.init n (fun _ -> draw ())) in
          let x = Contract_cases.cast_to d (A.Any made) in
          let back = Contract_cases.cast_to (D.Any D.Int64) x in
          let vs = A.to_array (A.expect D.Int64 back) in
          (x, Array.map (wrap form) vs))

(* Runs [c] and checks each output against the exact modular sum: each element
   enters the accumulator by its cast, the products and their sum wrap, init
   enters by its cast and adds, and the sum enters dst by its cast. Its
   launches. *)
let int_bits c =
  tiles_64 c.ifew @@ fun () ->
  let bk = backend () in
  let st = Random.State.make [| c.iseed |] in
  let bt = c.ibatch and m = c.im and n = c.in_ and k = c.ik in
  let form = int_form c.iacc in
  let full = c.ifull in
  let A.Any a, av = int_operand ~full st c.ia [ bt; m; k ] ~acc:c.iacc in
  let A.Any b, bv =
    if c.inn then
      let x, v = int_operand ~full st c.ib [ bt; n; k ] ~acc:c.iacc in
      (x, fun z kk j -> v.((((z * n) + j) * k) + kk))
    else
      let A.Any x, v = int_operand ~full st c.ib [ bt; k; n ] ~acc:c.iacc in
      (A.Any x, fun z kk j -> v.((((z * k) + kk) * n) + j))
  in
  let outputs = bt * m * n in
  let init =
    if not c.iinit then None
    else
      let x, v = int_operand st c.idst [ bt; m; n ] ~acc:c.iacc in
      Some (x, v)
  in
  let (D.Any ddt) = c.idst in
  let dst = A.create (S.device ()) ddt (Iarray.of_list [ bt; m; n ]) in
  let init_on =
    Option.map (fun (x, _) -> Support.on bk (A.expect ddt x)) init
  in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (2, if c.inn then 2 else 1) ])
      ~segments:None ~acc:c.iacc ~dst ~init:init_on
      { Nx_kernel.array = A.Any (Support.on bk a); scale = None }
      { Nx_kernel.array = A.Any (Support.on bk b); scale = None }
  in
  equal answer A.Done got;
  let sums =
    Array.init outputs (fun o ->
        let z = o / (m * n) and i = o / n mod m and j = o mod n in
        let s = ref 0L in
        (* At the maxima every term is the same product, and k of them sum to k
           times it modulo 2^64. *)
        if full && k > 0 then
          s := Int64.mul (Int64.of_int k) (Int64.mul av.(0) (bv 0 0 0))
        else
          for kk = 0 to k - 1 do
            s :=
              Int64.add !s (Int64.mul av.((((z * m) + i) * k) + kk) (bv z kk j))
          done;
        let s =
          match init with None -> !s | Some (_, v) -> Int64.add !s v.(o)
        in
        wrap form s)
  in
  let held =
    Contract_cases.cast_to c.iacc
      (A.Any (A.of_array D.Int64 (Iarray.of_list [ bt; m; n ]) sums))
  in
  let want = Contract_cases.cast_to c.idst held in
  if not (Support.same_bytes want (A.Any dst)) then
    equal (array string) (Support.elements want) (Support.elements (A.Any dst));
  launches

(* nx_metal.mli and nx_kernel.mli: an integer contraction is exact modulo its
   accumulator's width. Each element enters the accumulator by its cast, the
   products and their sum wrap, init enters by its cast and adds, and the sum
   enters dst by its cast: at every accumulator width, on the half units and off
   them, over blocks of 256 terms and their neighbours, through sums that
   wrap. *)
let law_int_bits c =
  let narrow = List.mem c.ia narrow_ints && List.mem c.ib narrow_ints in
  cover "the half units" (narrow && c.ik > 0);
  cover "other operands" (not narrow);
  cover "float operands"
    (List.exists (fun (D.Any d) -> D.is D.Float d) [ c.ia; c.ib ]);
  cover "a 64-bit accumulator" (fst (int_form c.iacc) = 64);
  cover "an 8-bit accumulator" (fst (int_form c.iacc) = 8);
  cover "a 4-bit accumulator" (fst (int_form c.iacc) = 4);
  cover "a dst of another dtype" (c.idst <> c.iacc);
  cover "about a block of 256" (List.mem c.ik [ 255; 256; 257; 513 ]);
  cover "no product" (c.ik = 0);
  cover "an init" c.iinit;
  let bool_operand = List.mem (D.Any D.Bool) [ c.ia; c.ib ] in
  cover "a boolean on the half units" (bool_operand && narrow && c.ik > 0);
  cover "a boolean off them" (bool_operand && (not narrow) && c.ik > 0);
  let launches = int_bits c in
  let module K = Nx_metal.Kernels in
  let ranges (t : Plan.traced) =
    if int_tile t.kernel then word t K.contract_ranges
    else if t.kernel = K.Contract_simt then word t K.simt_ranges
    else 0
  in
  cover "several ranges" (List.exists (fun t -> ranges t > 1) launches);
  (* nx_metal.mli: operands of 8 bits or fewer go on the half units. *)
  if c.ik > 0 then
    equal ~msg:"on the half units" bool narrow
      (List.exists (fun (t : Plan.traced) -> int_tile t.kernel) launches)

type iwindowed = { ic : ibits; ballowed : float }

let pp_iwindowed ppf w =
  Format.fprintf ppf "%a, launches of at most %gs" pp_ibits w.ic w.ballowed

(* Calls whose operands need no copy, whose launches of a few elements of
   several axes take more than the least budget: b is read n x k where it gives
   the tiles' rows. The least budget holds each call's smallest unit: a SIMT
   launch of one group over a chunk of 32 terms, at most 2 ranges of int64 by
   int64 for these extents, is estimated at 1.18e-5 s past its dispatch, and a
   budget below it leaves the call no launch. *)
let iwindowed_case =
  Gen.(
    with_pp pp_iwindowed
      (let+ ic = ibits_case
       and+ ballowed = of_list [ Plan.dispatch +. 1.5e-5; 3e-5; 1e-4 ] in
       let exchanged = C.exchanges ~m:ic.im ~n:ic.in_ in
       { ic = { ic with inn = ic.inn || exchanged }; ballowed }))

(* nx_metal.mli: no bit depends on the launches. Under budgets that cut an
   integer call's tiles or SIMT outputs into windows, or their ranges into
   chunks that each resume the 64-bit sums before them, every output keeps its
   exact value. *)
let law_int_window_bits w =
  let module K = Nx_metal.Kernels in
  let launches = bounded w.ballowed (fun () -> int_bits w.ic) in
  let summing (t : Plan.traced) =
    int_tile t.kernel || t.kernel = K.Contract_simt
  in
  let words (t : Plan.traced) =
    if int_tile t.kernel then (word t K.contract_first, word t K.contract_length)
    else (word t K.simt_first, word t K.simt_length)
  in
  let sums = List.filter summing launches in
  cover "the half units"
    (List.exists (fun (t : Plan.traced) -> int_tile t.kernel) sums);
  cover "SIMT"
    (List.exists (fun (t : Plan.traced) -> t.kernel = K.Contract_simt) sums);
  cover "several windows" (List.exists (fun t -> fst (words t) > 0) sums);
  cover "chunks of k" (List.exists (fun t -> snd (words t) > 0) sums)

let ibits_of ?(few = false) ?(inn = false) ?(iinit = false) ?idst iacc ia ib
    ibatch im in_ ik =
  let idst = Option.value ~default:iacc idst in
  {
    iacc;
    ia;
    ib;
    idst;
    ibatch;
    im;
    in_;
    ik;
    inn;
    iinit;
    ifull = false;
    iseed = 13;
    ifew = few;
  }

(* Calls that reach each tile on the half units, both layouts of b, the blocks'
   edges, each accumulator width and the other operands. *)
let ibits_examples =
  let open D in
  [
    ibits_of ~few:true (Any Int32) (Any Int8) (Any Int8) 1 256 256 256;
    ibits_of ~inn:true (Any Uint32) (Any Uint8) (Any Int8) 1 100 70 257;
    ibits_of (Any Int8) (Any Int8) (Any Uint8) 1 16 64 255;
    ibits_of ~iinit:true (Any Uint8) (Any Int4) (Any Uint4) 1 30 20 513;
    ibits_of ~idst:(Any Float32) (Any Int64) (Any Bool) (Any Int8) 2 9 33 600;
    ibits_of (Any Int4) (Any Bit) (Any Uint8) 1 20 20 40;
    ibits_of (Any Int64) (Any Int64) (Any Int64) 1 17 18 70;
    ibits_of ~iinit:true ~idst:(Any Bool) (Any Uint16) (Any Float32) (Any Int32)
      1 40 3 33;
    ibits_of ~inn:true (Any Uint64) (Any Bfloat16) (Any Uint64) 2 5 70 300;
    ibits_of ~iinit:true (Any Int16) (Any Int16) (Any Uint16) 1 8 8 0;
    ibits_of ~inn:true (Any Int32) (Any Int8) (Any Bool) 1 24 40 300;
    ibits_of ~iinit:true ~idst:(Any Bool) (Any Int32) (Any Bool) (Any Int16) 1
      20 30 40;
  ]
  @ List.map
      (fun (few, m, n) ->
        ibits_of ~few (Any Int32) (Any Int8) (Any Int8) 1 m n 40)
      column_edges

(* Integer calls cut into chunks whose last tile starts at an odd column. *)
let iwindowed_examples =
  let open D in
  [
    {
      ic = ibits_of (Any Int32) (Any Int8) (Any Int8) 1 16 129 4096;
      ballowed = 3e-5;
    };
    {
      ic = ibits_of (Any Int32) (Any Int8) (Any Int8) 1 40 47 4096;
      ballowed = 3e-5;
    };
    (* Outputs cut into windows, on the half units and by SIMT, which 30 draws
       can miss. *)
    {
      ic = ibits_of (Any Int32) (Any Int8) (Any Int8) 1 256 256 64;
      ballowed = Plan.dispatch +. 1.5e-5;
    };
    {
      ic = ibits_of (Any Int64) (Any Int64) (Any Int64) 1 64 64 300;
      ballowed = Plan.dispatch +. 1.5e-5;
    };
  ]

type imax = { mc : ibits; mallowed : float option }

let pp_imax ppf r =
  match r.mallowed with
  | None -> pp_ibits ppf r.mc
  | Some s -> Format.fprintf ppf "%a, launches of at most %gs" pp_ibits r.mc s

(* Calls whose operands sit at their maxima, where a block's float sums are
   exact only up to 2^24: 255 x 255 = 65,025 is odd, so 259 such terms already
   round, and 127 x 255 = 32,385 rounds past 518. On each tile and b's layout,
   at 256, 257, 512 and 513 terms, into int32 and uint32, which hold the sum,
   and uint8, where it wraps; then 127 x 255 at 1024 terms, and one call cut
   into chunks of k. *)
let imax_rows =
  let open D in
  let full ?(few = false) ?(inn = false) acc a b bt m n k =
    {
      mc = { (ibits_of ~few ~inn acc a b bt m n k) with ifull = true };
      mallowed = None;
    }
  in
  let accs = [| Any Int32; Any Uint32; Any Uint8 |] in
  let shapes = [ (true, 256, 256); (false, 100, 70); (false, 16, 64) ] in
  let at = ref 0 in
  let rows =
    List.concat_map
      (fun (few, m, n) ->
        List.concat_map
          (fun inn ->
            List.map
              (fun k ->
                let acc = accs.(!at mod Array.length accs) in
                incr at;
                full ~few ~inn acc (Any Uint8) (Any Uint8) 1 m n k)
              [ 256; 257; 512; 513 ])
          [ false; true ])
      shapes
  in
  let chunked = full (Any Uint32) (Any Uint8) (Any Uint8) 1 16 64 513 in
  rows
  @ [
      full (Any Int32) (Any Int8) (Any Uint8) 1 16 64 1024;
      full ~inn:true (Any Uint32) (Any Int8) (Any Uint8) 1 100 70 1024;
      { chunked with mallowed = Some (Plan.dispatch +. 1e-5) };
    ]

(* nx_metal.mli: an integer contraction is exact modulo its accumulator, so a
   block's sums stay exact at the operands' maxima, on every tile and layout,
   over one range of k or several, and resumed across chunks. *)
let law_int_maxima r =
  let module K = Nx_metal.Kernels in
  let launches =
    match r.mallowed with
    | None -> int_bits r.mc
    | Some s -> bounded s (fun () -> int_bits r.mc)
  in
  let tiles =
    List.filter (fun (t : Plan.traced) -> int_tile t.kernel) launches
  in
  equal ~msg:"on the half units" bool true (tiles <> []);
  List.iter
    (fun k ->
      cover (K.name k)
        (List.exists (fun (t : Plan.traced) -> t.kernel = k) tiles))
    K.
      [
        Contract_i8_64_kn;
        Contract_i8_32_kn;
        Contract_i8_16_kn;
        Contract_i8_64_nk;
        Contract_i8_32_nk;
        Contract_i8_16_nk;
      ];
  let ranges =
    List.fold_left (fun r t -> Int.max r (word t K.contract_ranges)) 0 tiles
  in
  cover "one range" (ranges = 1);
  cover "several ranges" (ranges > 1);
  cover "chunks of k"
    (List.exists (fun t -> word t K.contract_length > 0) tiles)

(* Float contractions into other dtypes *)

(* Every dtype a float32 contraction's dst, and so its init, may take: all but
   float64 and complex128. *)
let held_dtypes =
  D.
    [
      Any Float32;
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
      Any Int64;
      Any Uint64;
      Any Int32;
      Any Uint32;
      Any Int16;
      Any Uint16;
      Any Int8;
      Any Uint8;
      Any Int4;
      Any Uint4;
      Any Complex64;
      Any Bool;
      Any Bit;
    ]

type fheld = {
  finit : bool;
  fdst : D.any;
  fbatch : int;
  fm : int;
  fn : int;
  fk : int;
  fseed : int;
}

let pp_fheld ppf c =
  let name (D.Any d) = D.name d in
  Format.fprintf ppf "%d x %d x %d x %d float32 into %s%s, seed %d" c.fbatch
    c.fm c.fn c.fk (name c.fdst)
    (if c.finit then ", init" else "")
    c.fseed

(* Few outputs, so that some calls take several ranges, and k about the
   spans. *)
let fheld_case =
  let open Gen in
  let pick = of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) held_dtypes in
  with_pp pp_fheld
    (let+ finit = bool
     and+ fdst = pick
     and+ fbatch = frequency [ (3, constant 1); (1, constant 2) ]
     and+ fm = frequency [ (2, int_range 1 16); (1, int_range 17 70) ]
     and+ fn = frequency [ (2, int_range 1 16); (1, int_range 17 70) ]
     and+ fk =
       frequency
         [ (1, constant 0); (2, int_range 1 40); (2, int_range 300 700) ]
     and+ fseed = nat in
     { finit; fdst; fbatch; fm; fn; fk; fseed })

(* nx_metal.mli and nx_kernel.mli: a float32 contraction's init enters the
   accumulator by its cast, and its sum enters dst by its cast, at every dtype
   but float64 and complex128: the stated association of the products at
   float32, then init as float32 holds it, then the cast into dst. *)
let law_float_held c =
  let module K = Nx_metal.Kernels in
  let bk = backend () in
  let st = Random.State.make [| c.fseed |] in
  let bt = c.fbatch and m = c.fm and n = c.fn and k = c.fk in
  let name = "float32" in
  let draw n = drawn_values st ~name ~low:(-8) ~special:16 n in
  let a, av = stored D.Float32 [ bt; m; k ] (draw (bt * m * k)) in
  let b, bv = stored D.Float32 [ bt; k; n ] (draw (bt * k * n)) in
  let shape = [ bt; m; n ] in
  let acc = D.Any D.Int64 in
  let (D.Any ddt) = c.fdst in
  let init =
    if c.finit then Some (A.expect ddt (fst (int_operand st c.fdst shape ~acc)))
    else None
  in
  let init_f32 =
    Option.map
      (fun x ->
        A.to_array
          (A.expect D.Float32
             (Contract_cases.cast_to (D.Any D.Float32) (A.Any x))))
      init
  in
  let dst = A.create (S.device ()) ddt (Iarray.of_list shape) in
  let p =
    planned
      {
        C.batch = bt;
        m;
        n;
        k;
        stage = C.F32;
        nk = false;
        complex = false;
        source = 0;
      }
  in
  cover "an init" c.finit;
  cover "a dst narrower than a byte"
    (List.mem c.fdst D.[ Any Int4; Any Uint4; Any Bit; Any Float4_e2m1fn ]);
  cover "an integer dst"
    (List.mem c.fdst
       D.
         [
           Any Int64;
           Any Uint64;
           Any Int32;
           Any Uint32;
           Any Int16;
           Any Uint16;
           Any Int8;
           Any Uint8;
         ]);
  cover "a boolean dst" (c.fdst = D.Any D.Bool);
  cover "a complex64 dst" (c.fdst = D.Any D.Complex64);
  cover "a float8 dst"
    (List.mem c.fdst D.[ Any Float8_e4m3fn; Any Float8_e5m2 ]);
  cover "no product" (k = 0);
  cover "several ranges" (k > 0 && p.ranges > 1);
  let got =
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (2, 1) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst
      ~init:(Option.map (Support.on bk) init)
      { Nx_kernel.array = A.Any (Support.on bk a); scale = None }
      { Nx_kernel.array = A.Any (Support.on bk b); scale = None }
  in
  equal answer A.Done got;
  let sums =
    Array.init
      (bt * m * n)
      (fun o ->
        let z = o / (m * n) and i = o / n mod m and j = o mod n in
        let init = Option.map (fun v -> v.(o)) init_f32 in
        associated_at ~ranges:p.ranges ~k ~init av
          (((z * m) + i) * k)
          1 bv
          ((z * k * n) + j)
          n)
  in
  let held = A.Any (A.of_array D.Float32 (Iarray.of_list shape) sums) in
  let want = Contract_cases.cast_to c.fdst held in
  if not (Support.same_bytes want (A.Any dst)) then
    equal (array string) (Support.elements want) (Support.elements (A.Any dst))

let fheld_of ?(finit = false) fdst fbatch fm fn fk =
  { finit; fdst; fbatch; fm; fn; fk; fseed = 5 }

(* Calls of each kind of init and dst, with no product and several ranges. *)
let fheld_examples =
  let open D in
  [
    fheld_of (Any Int32) 1 20 30 40;
    fheld_of ~finit:true (Any Bit) 1 9 17 600;
    fheld_of ~finit:true (Any Bool) 2 5 5 0;
    fheld_of ~finit:true (Any Complex64) 1 30 8 333;
    fheld_of ~finit:true (Any Float8_e4m3fn) 1 16 16 32;
    fheld_of (Any Uint4) 1 3 70 500;
    fheld_of ~finit:true (Any Int4) 1 3 70 500;
  ]

(* Float32 contractions of operands of other dtypes *)

(* nx_metal.mli: the dtypes whose every value float16, and bfloat16, holds
   exactly; the tiles stage at float16 where both operands are of the first, at
   bfloat16 where both are of the second, else at float32. *)
let half_exact =
  D.
    [
      Any Float16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
      Any Int8;
      Any Uint8;
      Any Int4;
      Any Uint4;
      Any Bool;
      Any Bit;
    ]

let bfloat_exact =
  D.Any D.Bfloat16 :: List.filter (fun d -> d <> D.Any D.Float16) half_exact

let converted_stage da db =
  let both l = List.mem da l && List.mem db l in
  if both half_exact then C.F16 else if both bfloat_exact then C.Bf16 else C.F32

(* The index of number [t] in C order of the extents [dims], and the number of
   [idx]. *)
let index_of dims t =
  snd
    (List.fold_right (fun e (t, idx) -> (t / e, (t mod e) :: idx)) dims (t, []))

(* The position of [idx] in [dims]' C order, by direct recursion: it runs for
   every element of a reference's operands. *)
let number_of dims idx =
  let rec go a dims idx =
    match (dims, idx) with
    | e :: dims, i :: idx -> go ((a * e) + i) dims idx
    | [], [] -> a
    | _ -> invalid_arg "number_of"
  in
  go 0 dims idx

(* [f idx] at every index of [dims], in C order. *)
let tabulated dims f =
  let d = Array.of_list dims in
  let rank = Array.length d in
  let n = Array.fold_left ( * ) 1 d in
  let idx = Array.make rank 0 in
  Array.init n (fun t ->
      let rest = ref t in
      for x = rank - 1 downto 0 do
        idx.(x) <- !rest mod d.(x);
        rest := !rest / d.(x)
      done;
      f (Array.to_list idx))

(* The operands of a mixed call: pairs of any dtypes but float64 and complex128,
   but the pairs of one staged dtype. *)
let native = D.[ Any Float32; Any Float16; Any Bfloat16 ]

(* How a mixed call's operands lie on the device: fresh; both [skipped] elements
   into their buffers, an odd count, so that one narrower than a byte starts
   inside a byte; a or b broadcast along the batch; a stepped along m; b stepped
   along its major axis, k for k x n and n for n x k, which the tiles read in
   place, its inner axis staying unit; a flipped along m and b along its major
   axis; both batches stepped; in Rows calls, b's groups stepped; or b alone a
   count of elements into its buffer. *)
type cvview =
  | Fresh_operands
  | Offset_operands
  | Offset_b of int
  | Broadcast_a
  | Broadcast_b
  | Stepped_a
  | Stepped_b
  | Flipped
  | Stepped_batch
  | Stepped_groups

let skipped = 3

let cvview_name = function
  | Fresh_operands -> ""
  | Offset_operands -> Printf.sprintf ", operands %d elements in" skipped
  | Offset_b s -> Printf.sprintf ", b %d elements in" s
  | Broadcast_a -> ", a broadcast along the batch"
  | Broadcast_b -> ", b broadcast along the batch"
  | Stepped_a -> ", a stepped along m"
  | Stepped_b -> ", b stepped along its major axis"
  | Flipped -> ", a and b flipped along their major axes"
  | Stepped_batch -> ", batches stepped"
  | Stepped_groups -> ", b's groups stepped"

let cvview_case groups =
  let views =
    [
      Offset_operands;
      Broadcast_a;
      Broadcast_b;
      Stepped_a;
      Stepped_b;
      Flipped;
      Stepped_batch;
    ]
    @ if groups then [ Stepped_groups ] else []
  in
  Gen.frequency
    ((List.length views, Gen.constant Fresh_operands)
    :: List.map (fun v -> (1, Gen.constant v)) views)

(* The views of a's and b's axes [view] makes, [a_m] a's axis of rows and
   [b_major] b's major axis, the one before its inner. *)
let views_of view ~a_m ~b_major =
  match view with
  | Fresh_operands | Stepped_groups -> (`Fresh, `Fresh)
  | Offset_operands -> (`Skip skipped, `Skip skipped)
  | Offset_b s -> (`Fresh, `Skip s)
  | Broadcast_a -> (`Broadcast 0, `Fresh)
  | Broadcast_b -> (`Fresh, `Broadcast 0)
  | Stepped_a -> (`Step a_m, `Fresh)
  | Stepped_b -> (`Fresh, `Step b_major)
  | Flipped -> (`Flip a_m, `Flip b_major)
  | Stepped_batch -> (`Step 0, `Step 0)

type cvbits = {
  va : D.any;
  vb : D.any;
  vnk : bool;
  vat : bool;
  vview : cvview;
  vbatch : int;
  vm : int;
  vn : int;
  vk : int;
  vinit : bool;
  vdst : D.any;
  vlow : int;
  vseed : int;
}

let pp_cvbits ppf c =
  let name (D.Any d) = D.name d in
  Format.fprintf ppf
    "%d x %d x %d x %d, %s x %s%s%s%s into float32, values from 2^%d, %sdst \
     %s, seed %d"
    c.vbatch c.vm c.vn c.vk (name c.va) (name c.vb)
    (if c.vnk then ", b n x k" else "")
    (if c.vat then ", a strided" else "")
    (cvview_name c.vview) c.vlow
    (if c.vinit then "init, " else "")
    (name c.vdst) c.vseed

let cvbits_case =
  let open Gen in
  let pick = of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) held_dtypes in
  let rows =
    frequency
      [ (2, int_range 1 16); (2, int_range 17 80); (1, int_range 81 160) ]
  in
  let depth =
    frequency
      [
        (1, constant 0);
        (3, int_range 1 80);
        (2, int_range 81 600);
        (1, of_list [ 32; 40; 2048 ]);
      ]
  in
  with_pp pp_cvbits
    (let+ va = pick
     and+ vb = pick
     and+ vnk = bool
     and+ vat = frequency [ (3, constant false); (1, constant true) ]
     and+ vview = cvview_case false
     and+ vbatch = frequency [ (3, constant 1); (1, int_range 2 3) ]
     and+ vm = rows
     and+ vn = rows
     and+ vk = depth
     and+ vinit = bool
     and+ vdst = of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) native
     and+ vlow = frequency [ (3, constant (-8)); (1, constant (-71)) ]
     and+ vseed = nat in
     let vb = if va = vb && List.mem va native then D.Any D.Int8 else vb in
     { va; vb; vnk; vat; vview; vbatch; vm; vn; vk; vinit; vdst; vlow; vseed })

(* Whether a float8 code of [dt] is a NaN or an infinity. *)
let float8_special (type v s) (dt : (v, s) D.t) code =
  match dt with
  | D.Float8_e4m3fn -> code land 0x7f = 0x7f
  | _ -> code land 0x7c = 0x7c

(* A float8 code of [dt] from [st], each finite one alike likely, a NaN or an
   infinity once in about [special] codes. *)
let float8_code st dt ~special =
  let rec finite () =
    let c = Random.State.int st 256 in
    if float8_special dt c then finite () else c
  in
  if Random.State.int st special = 0 then
    let rec pick () =
      let c = Random.State.int st 256 in
      if float8_special dt c then c else pick ()
    in
    pick ()
  else finite ()

(* A host array of [dt] and [shape] from [st], and its values as float32 holds
   them, by the cast (nx_kernel.mli). A float8's and a float4's codes are drawn
   as bytes and nibbles read through [Nx_array.bitcast], so that every code
   arises, subnormals included, and a float8's NaN or infinity once in about
   [special] codes; another float's values as {!drawn_values}'s, a special once
   in about [special], from 2^[low] to 2^(low + 16); an integer's from 1 to
   2^bits, then cast, so that the extremes saturate; a complex64's parts both
   drawn; a boolean's bytes over 0..255. *)
let converted_operand ~special st (D.Any dt as d) shape ~low =
  let n = List.fold_left ( * ) 1 shape in
  let sh = Iarray.of_list shape in
  let floats name = drawn_values st ~name ~low ~special in
  let x =
    match dt with
    | D.Bool ->
        let bytes = Array.init n (fun _ -> bool_byte st) in
        A.Any (Option.get (A.bitcast D.Bool (A.of_array D.Uint8 sh bytes)))
    | D.Float8_e4m3fn | D.Float8_e5m2 ->
        let bytes = Array.init n (fun _ -> float8_code st dt ~special) in
        A.Any (Option.get (A.bitcast dt (A.of_array D.Uint8 sh bytes)))
    | D.Float4_e2m1fn ->
        let nibbles = Array.init n (fun _ -> Random.State.int st 16) in
        A.Any (Option.get (A.bitcast dt (A.of_array D.Uint4 sh nibbles)))
    | D.Complex64 ->
        A.Any (fst (stored_complex shape (floats "float32" (2 * n))))
    | _ -> (
        match D.kind dt with
        | D.Float ->
            let name = if List.mem d native then D.name dt else "float32" in
            Contract_cases.cast_to d
              (A.Any (A.of_array D.Float64 sh (floats name n)))
        | _ ->
            let bits = D.bits dt in
            let draw _ =
              if Random.State.int st 16 = 0 then
                pick_from st [| infinity; neg_infinity; nan; 0.; -1. |]
              else
                let v =
                  Float.ldexp
                    (1. +. Random.State.float st 1.)
                    (Random.State.int st bits)
                in
                if Random.State.bool st then v else -.v
            in
            Contract_cases.cast_to d
              (A.Any (A.of_array D.Float64 sh (Array.init n draw))))
  in
  let held = Contract_cases.cast_to (D.Any D.Float32) x in
  (x, A.to_array (A.expect D.Float32 held))

(* Covers of the float8 and float4 codes operands hold, [ops] their dtypes and
   float32 values: subnormals and NaN of each float8, e5m2's infinities, and all
   16 codes of a float4 operand. *)
let cover_codes ops =
  let any d f =
    List.exists (fun (d', vs) -> d' = D.Any d && Array.exists f vs) ops
  in
  let subnormal below v = v <> 0. && Float.abs v < below in
  cover "a float8_e4m3fn subnormal" (any D.Float8_e4m3fn (subnormal 0x1p-6));
  cover "a float8_e5m2 subnormal" (any D.Float8_e5m2 (subnormal 0x1p-14));
  cover "a float8_e4m3fn NaN" (any D.Float8_e4m3fn Float.is_nan);
  cover "a float8_e5m2 NaN" (any D.Float8_e5m2 Float.is_nan);
  cover "a float8_e5m2 infinity"
    (any D.Float8_e5m2 (fun v -> Float.abs v = infinity));
  let codes vs =
    List.length
      (List.sort_uniq compare
         (Array.to_list (Array.map Int64.bits_of_float vs)))
  in
  cover "every float4 code"
    (List.exists
       (fun (d, vs) -> d = D.Any D.Float4_e2m1fn && codes vs = 16)
       ops)

(* Covers of the integers operands hold, [ops] their dtypes and float32 values:
   past 2^24, where float32 rounds, an int64 at its extremes, and a uint64 past
   2^63. *)
let cover_wide ops =
  let any ds f =
    List.exists (fun (d, vs) -> List.mem d ds && Array.exists f vs) ops
  in
  let ints = D.[ Any Int64; Any Uint64; Any Int32; Any Uint32 ] in
  cover "an integer past 2^24" (any ints (fun v -> Float.abs v > 0x1p24));
  cover "an int64 at an extreme"
    (any [ D.Any D.Int64 ] (fun v -> Float.abs v = 0x1p63));
  cover "a uint64 past 2^63" (any [ D.Any D.Uint64 ] (fun v -> v > 0x1p63))

(* Covers of the outputs a float8 operand reaches: [want o] is output [o] as dst
   holds it, [dropped o kk] the same with term [kk] made +0, [init o] its init
   (+0 without), [terms o] its terms' operand values, of dtypes [da] and [db]. A
   finite output over several [ranges] that a float8 subnormal term changes,
   which only a finite decode of it gives; a NaN from one float8 NaN term among
   finite ones; an infinity from an e5m2 infinity. *)
let cover_outputs ~da ~db ~ranges ~outputs ~want ~dropped ~init ~terms =
  let is d = da = D.Any d || db = D.Any d in
  let f8 = is D.Float8_e4m3fn || is D.Float8_e5m2 in
  let subnormal d v =
    v <> 0.
    &&
    match d with
    | D.Any D.Float8_e4m3fn -> Float.abs v < 0x1p-6
    | D.Any D.Float8_e5m2 -> Float.abs v < 0x1p-14
    | _ -> false
  in
  (* Whether some output [o] with a finite init and a value [value] holds has
     terms [f o] holds. *)
  let some value f =
    let found = ref false and o = ref 0 in
    while (not !found) && !o < outputs do
      found := Float.is_finite (init !o) && value (want !o) && f !o (terms !o);
      incr o
    done;
    !found
  in
  let changes o kk =
    Int64.bits_of_float (dropped o kk) <> Int64.bits_of_float (want o)
  in
  (* Whether output [o]'s float8 subnormal term of the largest product changes
     it: one more sum an output, so the cover costs no more than the law's own
     sums. *)
  let subnormal_term o ts =
    let best = ref (-1) and most = ref 0. in
    Array.iteri
      (fun kk (a, b) ->
        let p = Float.abs (a *. b) in
        if (subnormal da a || subnormal db b) && p > !most then begin
          best := kk;
          most := p
        end)
      ts;
    !best >= 0 && changes o !best
  in
  cover
    "a finite output over several ranges that a float8 subnormal term changes"
    (f8 && ranges > 1 && some Float.is_finite subnormal_term);
  let nan_of d v =
    Float.is_nan v && (d = D.Any D.Float8_e4m3fn || d = D.Any D.Float8_e5m2)
  in
  let finite (a, b) =
    (Float.is_finite a || nan_of da a) && (Float.is_finite b || nan_of db b)
  in
  let one_nan ts =
    let nans =
      Array.fold_left
        (fun c (a, b) -> if nan_of da a || nan_of db b then c + 1 else c)
        0 ts
    in
    nans = 1 && Array.for_all finite ts
  in
  cover "a NaN output from one float8 NaN term"
    (f8 && some Float.is_nan (fun _ ts -> one_nan ts));
  let inf_of d v = d = D.Any D.Float8_e5m2 && Float.abs v = infinity in
  cover "an infinite output from a float8_e5m2 infinity"
    (is D.Float8_e5m2
    && some
         (fun v -> Float.abs v = infinity)
         (fun _ -> Array.exists (fun (a, b) -> inf_of da a || inf_of db b)))

(* An operand of [d] and logical [shape] drawn from [st] and made on [bk] as
   [view] says: fresh, [s] elements into its buffer, broadcast along axis [ax]
   from one element, every other element of axis [ax] of a buffer twice as long
   there, or axis [ax] reversed. Its float32 values in C order of [shape]. *)
let laid_operand ~special st bk d shape ~low view =
  let moved mv (A.Any x) = A.Any (Option.get (A.move mv x)) in
  let range start count step = { M.start; count; step } in
  let total = List.fold_left ( * ) 1 shape in
  let resized ax f = List.mapi (fun i e -> if i = ax then f e else e) shape in
  let at_axis ax j idx = List.mapi (fun i x -> if i = ax then j else x) idx in
  match view with
  | `Fresh ->
      let x, v = converted_operand ~special st d shape ~low in
      (Contract_cases.on bk x, v)
  | `Skip s ->
      let x, v = converted_operand ~special st d [ s + total ] ~low in
      let x = moved (M.Slice [| range s total 1 |]) (Contract_cases.on bk x) in
      (moved (M.Reshape (Iarray.of_list shape)) x, Array.sub v s total)
  | `Broadcast ax ->
      let stored = resized ax (fun _ -> 1) in
      let x, v = converted_operand ~special st d stored ~low in
      let x =
        moved (M.Broadcast (Iarray.of_list shape)) (Contract_cases.on bk x)
      in
      ( x,
        Array.init total (fun t ->
            v.(number_of stored (at_axis ax 0 (index_of shape t)))) )
  | `Step ax ->
      let stored = resized ax (fun e -> 2 * e) in
      let x, v = converted_operand ~special st d stored ~low in
      let keep =
        List.mapi (fun i e -> range 0 e (if i = ax then 2 else 1)) shape
      in
      let x = moved (M.Slice (Iarray.of_list keep)) (Contract_cases.on bk x) in
      let doubled idx = at_axis ax (2 * List.nth idx ax) idx in
      ( x,
        Array.init total (fun t ->
            v.(number_of stored (doubled (index_of shape t)))) )
  | `Flip ax ->
      let x, v = converted_operand ~special st d shape ~low in
      let e = List.nth shape ax in
      let reversed i e = if i = ax then range (e - 1) e (-1) else range 0 e 1 in
      let keep = List.mapi reversed shape in
      let x = moved (M.Slice (Iarray.of_list keep)) (Contract_cases.on bk x) in
      let flipped idx = at_axis ax (e - 1 - List.nth idx ax) idx in
      ( x,
        Array.init total (fun t ->
            v.(number_of shape (flipped (index_of shape t)))) )

(* An unsegmented mixed call's a and b on [bk], drawn from [st] as [c] says, and
   their float32 values at a batch element, a row or term, and a term or
   column. *)
let cv_operands st bk c =
  let bt = c.vbatch and m = c.vm and n = c.vn and k = c.vk in
  let low = c.vlow in
  let a_view, b_view = views_of c.vview ~a_m:1 ~b_major:1 in
  (* A special once in about 4k values leaves most outputs of k terms finite. *)
  let special = Int.max 64 (4 * k) in
  let a, av =
    if c.vat then
      let skip = if c.vview = Offset_operands then `Skip skipped else `Fresh in
      let x, v = laid_operand ~special st bk c.va [ bt; k; m ] ~low skip in
      let (A.Any x) = x in
      let x =
        A.Any (Option.get (A.move (M.Permute (Iarray.of_list [ 0; 2; 1 ])) x))
      in
      (x, fun z i kk -> v.((((z * k) + kk) * m) + i))
    else
      let x, v = laid_operand ~special st bk c.va [ bt; m; k ] ~low a_view in
      (x, fun z i kk -> v.((((z * m) + i) * k) + kk))
  in
  let b, bv =
    if c.vnk then
      let x, v = laid_operand ~special st bk c.vb [ bt; n; k ] ~low b_view in
      (x, fun z kk j -> v.((((z * n) + j) * k) + kk))
    else
      let x, v = laid_operand ~special st bk c.vb [ bt; k; n ] ~low b_view in
      (x, fun z kk j -> v.((((z * k) + kk) * n) + j))
  in
  (a, av, b, bv)

(* The bytes an element of [x] takes, 0 below a byte, and whether [x] starts
   inside a byte. *)
let width (A.Any x) = D.bits (A.dtype x) / 8

let phased (A.Any x) =
  let bits = D.bits (A.dtype x) in
  bits < 8 && A.Layout.offset (A.layout x) * bits mod 8 <> 0

(* Covers of the views of a and b that the tiles read in place, as none of
   [launches] copies an operand: an odd offset at widths 1, 2 and 8, and at a
   width other than [stage]'s; a or b broadcast along the batch; a stepped along
   [a_m]; b stepped along [b_major]; a flipped along [a_m] and b along
   [b_major]; a stepped batch. And a start inside a byte, which the widening
   reads. *)
let cover_layouts ~a ~b ~a_m ~b_major ~stage launches =
  let copied =
    List.exists (fun (t : Plan.traced) -> apart_copy t.kernel) launches
  in
  let stage_width =
    match stage with C.I8 -> 1 | C.F16 | C.Bf16 -> 2 | C.F32 -> 4
  in
  let read (A.Any x) = (not copied) && D.bits (A.dtype x) >= 8 in
  let layout (A.Any x) = A.layout x in
  let odd x = read x && A.Layout.offset (layout x) mod 2 = 1 in
  List.iter
    (fun w ->
      cover
        (Printf.sprintf "an operand of %d bytes at an odd offset, in place" w)
        (List.exists (fun x -> odd x && width x = w) [ a; b ]))
    [ 1; 2; 8 ];
  cover "an odd offset at a width other than the stage's, in place"
    (List.exists (fun x -> odd x && width x <> stage_width) [ a; b ]);
  cover "an operand narrower than a byte starting inside a byte"
    (phased a || phased b);
  let extent x ax = Iarray.get (A.Layout.shape (layout x)) ax in
  let stride x ax = A.Layout.stride (layout x) ax in
  let dense x ax =
    Iarray.get
      (A.Layout.strides (A.Layout.contiguous (A.Layout.shape (layout x))))
      ax
  in
  let broadcast x = read x && extent x 0 > 1 && stride x 0 = 0 in
  let stepped x ax =
    read x && extent x ax > 1 && stride x ax = 2 * dense x ax
  in
  let flipped x ax = read x && extent x ax > 1 && stride x ax < 0 in
  cover "a broadcast along the batch, in place" (broadcast a);
  cover "b broadcast along the batch, in place" (broadcast b);
  cover "a stepped along m, in place" (stepped a a_m);
  cover "b stepped along its major axis, in place" (stepped b b_major);
  cover "a flipped along m, in place" (flipped a a_m);
  cover "b flipped along its major axis, in place" (flipped b b_major);
  cover "a stepped batch, in place" (stepped a 0 || stepped b 0)

(* Loaders by name *)

(* Whether [k] is a loader: a tile made for its operands' dtypes. *)
let is_loader k =
  List.exists
    (fun prefix -> String.starts_with ~prefix (Nx_metal.Kernels.name k))
    [ "contract_ld_"; "contract_rows_ld_" ]

(* The loaders of a dense or a grouped tile ([rows]), of an unscaled or a scaled
   b ([scaled]), by their names, but the banded ones. *)
let loaders ~rows ~scaled =
  let prefix = if rows then "contract_rows_ld_" else "contract_ld_" in
  let by name =
    let n = String.length name in
    let rec go i = i + 4 <= n && (String.sub name i 4 = "_by_" || go (i + 1)) in
    go 0
  in
  List.filter
    (fun k ->
      let name = Nx_metal.Kernels.name k in
      String.starts_with ~prefix name && by name = scaled && not (banded_twin k))
    (Array.to_list Nx_metal.Kernels.all)

(* nx_metal.mli: a call that takes a loader stages its operands through loaders
   alone ([keyed]); any other call through none. *)
let check_loaders ~keyed launches =
  let staging k = is_tile k || grouped_tile k || int_tile k || is_loader k in
  let names p =
    List.filter_map
      (fun (t : Plan.traced) ->
        if staging t.kernel && p t.kernel then
          Some (Nx_metal.Kernels.name t.kernel)
        else None)
      launches
  in
  if keyed then equal (list string) [] (names (fun k -> not (is_loader k)))
  else equal (list string) [] (names is_loader)

(* Covers each of [ks] launched. *)
let cover_kernels ks launches =
  List.iter
    (fun kn ->
      cover (Nx_metal.Kernels.name kn)
        (List.exists (fun (t : Plan.traced) -> t.kernel = kn) launches))
    ks

(* nx_metal.mli and nx_kernel.mli: a float32 contraction's operands of any
   dtypes but float64 and complex128 enter the accumulator by their casts: every
   output has the bits of the stated association of the cast operands' products
   at float32, the plan's of the stage its dtypes give, then init, rounded once
   into dst. *)
let law_converted_bits c =
  let bk = backend () in
  let st = Random.State.make [| c.vseed |] in
  let bt = c.vbatch and m = c.vm and n = c.vn and k = c.vk in
  let low = c.vlow in
  let a, av, b, bv = cv_operands st bk c in
  let shape = [ bt; m; n ] in
  let outputs = bt * m * n in
  let (D.Any ddt) = c.vdst in
  let init =
    if not c.vinit then None
    else
      let name = D.name ddt in
      let vs = drawn_values st ~name ~low ~special:16 outputs in
      let made = A.of_array D.Float64 (Iarray.of_list shape) vs in
      let x = A.expect ddt (Contract_cases.cast_to c.vdst (A.Any made)) in
      let held = Contract_cases.cast_to (D.Any D.Float32) (A.Any x) in
      Some (x, A.to_array (A.expect D.Float32 held))
  in
  let dst = A.create (S.device ()) ddt (Iarray.of_list shape) in
  let stage = converted_stage c.va c.vb in
  (* contract.mli: the bytes the tiles read change the launches alone. *)
  let p =
    planned
      { C.batch = bt; m; n; k; stage; nk = c.vnk; complex = false; source = 0 }
  in
  cover "float16" (stage = C.F16);
  cover "bfloat16" (stage = C.Bf16);
  cover "float32" (stage = C.F32);
  cover "a complex64 operand" (List.mem (D.Any D.Complex64) [ c.va; c.vb ]);
  cover "a 64-bit integer operand"
    (List.exists
       (fun d -> List.mem d [ c.va; c.vb ])
       D.[ Any Int64; Any Uint64 ]);
  cover "an operand narrower than a byte"
    (List.exists
       (fun d -> List.mem d [ c.va; c.vb ])
       D.[ Any Int4; Any Uint4; Any Bit; Any Float4_e2m1fn ]);
  cover "a float8 operand"
    (List.exists
       (fun d -> List.mem d [ c.va; c.vb ])
       D.[ Any Float8_e4m3fn; Any Float8_e5m2 ]);
  cover "exchanged" p.exchanged;
  cover "several ranges" (k > 0 && p.ranges > 1);
  cover "a partial step" (k mod C.depth stage p.tile <> 0);
  cover "an init" c.vinit;
  cover "no product" (k = 0);
  let ref_a, ref_b, sums, sum, terms =
    reference ~ranges:p.ranges ~bt ~m ~n ~k ~init:(Option.map snd init) av bv
  in
  cover_codes [ (c.va, ref_a); (c.vb, ref_b) ];
  cover_wide [ (c.va, ref_a); (c.vb, ref_b) ];
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (2, if c.vnk then 2 else 1) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
      { Nx_kernel.array = Contract_cases.on bk a; scale = None }
      { Nx_kernel.array = Contract_cases.on bk b; scale = None }
  in
  equal answer A.Done got;
  if not c.vat then cover_layouts ~a ~b ~a_m:1 ~b_major:1 ~stage launches;
  check_loaders ~keyed:false launches;
  cover "into dst"
    (k > 0
    && not
         (List.exists
            (fun (t : Plan.traced) ->
              t.kernel = Nx_metal.Kernels.Contract_finish)
            launches));
  let held = A.Any (A.of_array D.Float32 (Iarray.of_list shape) sums) in
  let as64 x =
    A.to_array (A.expect D.Float64 (Contract_cases.cast_to (D.Any D.Float64) x))
  in
  let want = as64 (Contract_cases.cast_to c.vdst held) in
  (* Output [o] with term [kk] made +0, as dst holds it. *)
  let dropped o kk =
    let v = sum ~drop:kk o in
    (as64
       (Contract_cases.cast_to c.vdst
          (A.Any (A.of_array D.Float32 [| 1 |] [| v |])))).(0)
  in
  cover_outputs ~da:c.va ~db:c.vb ~ranges:p.ranges ~outputs
    ~want:(fun o -> want.(o))
    ~dropped
    ~init:(fun o -> Option.fold ~none:0. ~some:(fun (_, v) -> v.(o)) init)
    ~terms;
  equal (array float_exact) want (as64 (A.Any (Support.host dst)))

let cvbits_of ?(vnk = false) ?(vat = false) ?(vview = Fresh_operands)
    ?(vinit = false) ?(vdst = D.Any D.Float32) va vb vbatch vm vn vk =
  {
    va;
    vb;
    vnk;
    vat;
    vview;
    vbatch;
    vm;
    vn;
    vk;
    vinit;
    vdst;
    vlow = -8;
    vseed = 9;
  }

(* Mixed calls that reach each tile, at each stage and layout of b; a complex64,
   a 64-bit, a widened and a float8 operand; float16 beside bfloat16; the
   exchange; several ranges; an init; no product; a dst of the stage, which the
   tiles store into; each view of the operands; and columns about a tile's
   edges. *)
let cvbits_examples =
  let open D in
  List.concat_map
    (fun (va, vb) ->
      List.concat_map
        (fun vnk ->
          [
            cvbits_of ~vnk va vb 6 256 256 40;
            cvbits_of ~vnk va vb 1 100 128 40;
            cvbits_of ~vnk va vb 1 16 64 40;
          ])
        [ false; true ])
    [
      (Any Float8_e4m3fn, Any Float16);
      (Any Int8, Any Bfloat16);
      (Any Int32, Any Float32);
    ]
  @ [
      cvbits_of (Any Complex64) (Any Float32) 1 30 40 100;
      cvbits_of ~vnk:true (Any Int64) (Any Uint64) 1 20 20 50;
      cvbits_of ~vat:true (Any Bit) (Any Float4_e2m1fn) 2 40 70 90;
      cvbits_of (Any Float16) (Any Bfloat16) 1 64 64 64;
      cvbits_of (Any Int4) (Any Bfloat16) 1 200 9 100;
      cvbits_of (Any Uint8) (Any Float8_e5m2) 1 16 64 2048;
      cvbits_of (Any Float8_e4m3fn) (Any Float16) 1 5 15 40;
      cvbits_of (Any Float8_e4m3fn) (Any Float16) 1 5 129 40;
      cvbits_of (Any Int8) (Any Float32) 1 40 47 40;
      cvbits_of ~vinit:true ~vdst:(Any Bfloat16) (Any Float32) (Any Int16) 1 30
        30 33;
      cvbits_of ~vinit:true (Any Bool) (Any Uint16) 2 5 5 0;
      cvbits_of ~vdst:(Any Float16) (Any Int8) (Any Float16) 1 64 64 64;
      cvbits_of ~vview:Offset_operands (Any Int8) (Any Float16) 1 40 64 70;
      cvbits_of ~vview:Offset_operands (Any Int64) (Any Float32) 2 30 40 50;
      cvbits_of ~vview:Offset_operands ~vnk:true (Any Float16) (Any Int4) 1 20
        64 64;
      cvbits_of ~vview:Broadcast_b (Any Float8_e4m3fn) (Any Float16) 3 64 64 40;
      cvbits_of ~vview:Offset_operands (Any Int16) (Any Float32) 1 30 40 50;
      cvbits_of ~vview:Offset_operands (Any Float16) (Any Float32) 2 30 40 64;
      cvbits_of ~vview:Broadcast_a (Any Int8) (Any Float16) 3 40 64 50;
      cvbits_of ~vview:Stepped_a (Any Float8_e4m3fn) (Any Float16) 2 40 70 50;
      cvbits_of ~vview:Stepped_a (Any Int64) (Any Float32) 1 30 70 64;
      cvbits_of ~vview:Stepped_b (Any Int8) (Any Bfloat16) 2 40 70 50;
      cvbits_of ~vview:Stepped_b ~vnk:true (Any Float32) (Any Uint8) 1 30 64 40;
      cvbits_of ~vview:Flipped (Any Int8) (Any Float16) 2 40 64 50;
      cvbits_of ~vview:Flipped ~vnk:true (Any Float32) (Any Int16) 1 30 40 64;
      cvbits_of ~vview:Stepped_batch (Any Uint8) (Any Bfloat16) 3 30 40 50;
      cvbits_of (Any Float8_e5m2) (Any Float16) 1 160 160 40;
      cvbits_of (Any Float8_e4m3fn) (Any Float16) 1 20 20 2048;
    ]

(* Rows segments *)

(* How a Rows call's ids are drawn: each in [0, g); those sorted; all of one
   group; of two groups, leaving the others empty; or about half of them outside
   [0, g), at -1, g, the int64 extremes, and values whose low 32 bits name a
   group. *)
type drawn_ids = Each | Sorted | One | Two | Outside

let ids_name = function
  | Each -> "each drawn"
  | Sorted -> "sorted"
  | One -> "one group"
  | Two -> "two groups"
  | Outside -> "some outside [0, g)"

(* How a Rows call's ids are viewed: fresh; 3 elements into their buffer; every
   other element of a row; one batch element's broadcast along the batch; or of
   a batch of 2 x 2 whose axes do not merge, so that the call copies them. *)
type ids_view =
  | Fresh_ids
  | Offset_ids
  | Stepped_ids
  | Broadcast_ids
  | Split_ids

let ids_view_name = function
  | Fresh_ids -> ""
  | Offset_ids -> " at an offset"
  | Stepped_ids -> " stepped"
  | Broadcast_ids -> " broadcast"
  | Split_ids -> " over a 2 x 2 batch, not merging"

type rbits = {
  rstage : int;
  rnk : bool;
  rbatch : int;
  rm : int;
  rsplit : bool;
  rn : int;
  rk : int;
  rg : int;
  rids : drawn_ids;
  rview : ids_view;
  rinit : bool;
  rint : bool;
  rseed : int;
}

let pp_rbits ppf c =
  Format.fprintf ppf
    "%d x %d%s x %d x %d in %d groups, %s, b %s, ids %s%s%s, seed %d" c.rbatch
    c.rm
    (if c.rsplit then " x 3" else "")
    c.rn c.rk c.rg
    (if c.rint then "int8 by int32 into int32"
     else if c.rstage = 3 then "complex64"
     else fst (List.nth stage_dts c.rstage))
    (if c.rnk then "n x k" else "k x n")
    (ids_name c.rids) (ids_view_name c.rview)
    (if c.rinit then ", init" else "")
    c.rseed

(* The most elements of b and the most terms of all outputs a draw takes, so
   that the host's reference stays short. *)
let rows_elements = 1 lsl 21
let rows_terms = 1 lsl 22

(* Rows below, at and above 16 g; g of none, one, many, past m, and of several
   of the prepass's chunks of 2048 buckets. *)
let rbits_case =
  let open Gen in
  let rows =
    frequency
      [ (2, int_range 1 16); (2, int_range 17 80); (1, int_range 81 300) ]
  in
  let groups =
    frequency
      [
        (1, constant 0);
        (1, constant 1);
        (4, int_range 2 40);
        (1, of_list [ 384; 1100; 4100; 5000 ]);
      ]
  in
  let depth =
    frequency
      [
        (1, constant 0);
        (3, int_range 1 80);
        (2, int_range 81 300);
        (1, of_list [ 33; 2048 ]);
      ]
  in
  with_pp pp_rbits
    (let+ rstage = int_range 0 3
     and+ rnk = bool
     and+ rbatch = frequency [ (3, constant 1); (1, constant 2) ]
     and+ rm = rows
     and+ rsplit = frequency [ (3, constant false); (1, constant true) ]
     and+ rn = frequency [ (3, int_range 1 80); (1, int_range 81 140) ]
     and+ rk = depth
     and+ rg = groups
     and+ rids = of_list [ Each; Sorted; One; Two; Outside ]
     and+ rview =
       frequency
         [
           (4, constant Fresh_ids);
           (1, constant Offset_ids);
           (1, constant Stepped_ids);
           (1, constant Broadcast_ids);
           (1, constant Split_ids);
         ]
     and+ rinit = bool
     and+ rint = frequency [ (4, constant false); (1, constant true) ]
     and+ rseed = nat in
     let rbatch =
       match rview with Split_ids -> 4 | Broadcast_ids -> 2 | _ -> rbatch
     in
     let gk = rbatch * Int.max 1 rg * Int.max 1 rk in
     let rn = Int.max 1 (Int.min rn (rows_elements / gk)) in
     let q = if rsplit then 3 else 1 in
     let rm =
       Int.max 1 (Int.min rm (rows_terms / (rbatch * q * rn * Int.max 1 rk)))
     in
     {
       rstage;
       rnk;
       rbatch;
       rm;
       rsplit;
       rn;
       rk;
       rg;
       rids;
       rview;
       rinit;
       rint;
       rseed;
     })

(* Values outside [0, g) for a group count [g]: -1, g, the int64 extremes, and
   values whose low 32 bits name group 1 or a drawn one. *)
let outside st g =
  let low = Int64.of_int (Random.State.int st (Int.max g 1)) in
  pick_from st
    [|
      -1L;
      Int64.of_int g;
      Int64.min_int;
      Int64.max_int;
      0x1_0000_0001L;
      -0xFFFF_FFFFL;
      Int64.add 0x1_0000_0000L low;
    |]

(* [bt] rows of [m] ids of [g] groups, drawn as [how] says, all one row's where
   [broadcast]. *)
let ids_drawn st how ~g ~m ~bt ~broadcast =
  let any () = Int64.of_int (Random.State.int st (Int.max g 1)) in
  let row () =
    match how with
    | Each -> Array.init m (fun _ -> any ())
    | Sorted ->
        let r = Array.init m (fun _ -> any ()) in
        Array.sort Int64.compare r;
        r
    | One -> Array.make m (any ())
    | Two ->
        let x = any () and y = any () in
        Array.init m (fun _ -> if Random.State.bool st then x else y)
    | Outside ->
        Array.init m (fun _ ->
            if Random.State.bool st then outside st g else any ())
  in
  if broadcast then
    let r = row () in
    Array.concat (List.init bt (fun _ -> r))
  else Array.concat (List.init bt (fun _ -> row ()))

(* The ids of [c]: [batch] rows of [m], drawn as [c.rids] says, all one row's
   where they are broadcast. *)
let rows_ids st c =
  ids_drawn st c.rids ~g:c.rg ~m:c.rm ~bt:c.rbatch
    ~broadcast:(c.rview = Broadcast_ids)

let in_group g id =
  Int64.compare id 0L >= 0 && Int64.compare id (Int64.of_int g) < 0

(* Whether [c] is a complex64 call: stage 3, of no integers. *)
let rows_complex c = c.rstage = 3 && not c.rint

(* The plan of [c]'s call: of [m] or [3 m] rows a batch element, a complex64
   call's real form of twice its columns and terms. *)
let rows_plan ?stage c =
  let complex = Option.is_none stage && rows_complex c in
  let stage =
    match stage with
    | Some s -> s
    | None -> List.nth [ C.F32; C.F16; C.Bf16; C.F32 ] c.rstage
  in
  let rows = if c.rsplit then 3 * c.rm else c.rm in
  let parts = if complex then 2 else 1 in
  let e =
    {
      C.batch = c.rbatch;
      m = rows;
      n = parts * c.rn;
      k = parts * c.rk;
      stage;
      nk = c.rnk;
      complex;
      source = 0;
    }
  in
  match C.grouped e ~groups:c.rg with
  | Some p -> p
  | None -> failf "%a has no plan" pp_extents e

(* [bt] rows of [m] ids in C order, as an array viewed as [view] says, of [bt;
   m] or, split, [2; 2; m], the view made of what [on] moved: a copy to another
   device packs the elements a view skips ({!Nx_array.to_device}). *)
let ids_viewed ~on view ~bt ~m ids =
  let shape l = Iarray.of_list l in
  let moved mv x = Option.get (A.move mv x) in
  let range start count step = { M.start; count; step } in
  match view with
  | Fresh_ids -> on (A.of_array D.Int64 (shape [ bt; m ]) ids)
  | Offset_ids ->
      let skip = 3 and n = bt * m in
      let x =
        on
          (A.of_array D.Int64
             (shape [ skip + n ])
             (Array.append (Array.make skip 0L) ids))
      in
      moved
        (M.Reshape (shape [ bt; m ]))
        (moved (M.Slice [| range skip n 1 |]) x)
  | Stepped_ids ->
      let x =
        A.of_array D.Int64
          (shape [ bt; 2 * m ])
          (Array.init
             (2 * bt * m)
             (fun e -> if e mod 2 = 0 then ids.(e / 2) else -1L))
      in
      moved (M.Slice [| range 0 bt 1; range 0 m 2 |]) (on x)
  | Broadcast_ids ->
      let x = on (A.of_array D.Int64 (shape [ 1; m ]) (Array.sub ids 0 m)) in
      moved (M.Broadcast (shape [ bt; m ])) x
  | Split_ids ->
      (* Stored [z2; z1; m], read as [z1; z2; m]: element (z1, z2, i) is batch
         element 2 z1 + z2's row i. *)
      let x =
        A.of_array D.Int64
          (shape [ 2; 2; m ])
          (Array.init (4 * m) (fun e ->
               let z2 = e / (2 * m) and z1 = e / m mod 2 and i = e mod m in
               ids.((((2 * z1) + z2) * m) + i)))
      in
      moved (M.Permute (shape [ 1; 0; 2 ])) (on x)

(* [c]'s ids as {!ids_viewed} views them. *)
let viewed_ids ~on c ids = ids_viewed ~on c.rview ~bt:c.rbatch ~m:c.rm ids

(* Runs [c]: a of [batch; m; k], or [batch; m; 3; k] segmented along m; b of
   [batch; g; k; n] or [batch; g; n; k]; ids of [batch; m]; dst starting as NaN,
   or -1 for integers, so that an unwritten output fails. Split ids take every
   operand's batch as [2; 2]. Its launches. *)
let rows_call ?sa ?sb c ~a ~b ~init ~dst ~ids =
  let bk = backend () in
  let split = c.rview = Split_ids in
  let lead = if split then 1 else 0 in
  let ra = lead + if c.rsplit then 3 else 2 in
  let rb = lead + if c.rnk then 3 else 2 in
  (* An operand's batch axis as two of 2 x 2, in C order. *)
  let batched (A.Any x) =
    if not split then A.Any x
    else
      let s = Iarray.to_list (A.Layout.shape (A.layout x)) in
      A.Any
        (Option.get
           (A.move (M.Reshape (Iarray.of_list (2 :: 2 :: List.tl s))) x))
  in
  let ids = viewed_ids ~on:(Support.on bk) c ids in
  let a = batched (A.Any (Support.on bk a))
  and b = batched (A.Any (Support.on bk b)) in
  let (A.Any dst) = batched (A.Any dst) in
  let init =
    Option.map
      (fun x ->
        let (A.Any y) = batched (A.Any x) in
        A.expect (A.dtype dst) (A.Any y))
      init
  in
  let batch = if split then [ (0, 0); (1, 1) ] else [ (0, 0) ] in
  Plan.traced @@ fun () ->
  Nx_metal.contract ~batch:(Iarray.of_list batch)
    ~contracting:(Iarray.of_list [ (ra, rb) ])
    ~segments:
      (Some (Nx_kernel.Rows { ids; a_axis = lead + 1; b_axis = lead + 1 }))
    ~acc:
      (if c.rint then D.Any D.Int32
       else if rows_complex c then D.Any D.Complex64
       else D.Any D.Float32)
    ~dst ~init
    { Nx_kernel.array = a; scale = sa }
    { Nx_kernel.array = b; scale = sb }

let rows_shapes c =
  let a =
    if c.rsplit then [ c.rbatch; c.rm; 3; c.rk ] else [ c.rbatch; c.rm; c.rk ]
  in
  let b =
    if c.rnk then [ c.rbatch; c.rg; c.rn; c.rk ]
    else [ c.rbatch; c.rg; c.rk; c.rn ]
  in
  let y =
    if c.rsplit then [ c.rbatch; c.rm; 3; c.rn ] else [ c.rbatch; c.rm; c.rn ]
  in
  (a, b, y)

(* b's element of group [id], term [kk] and column [j] at batch element [z], in
   the flat array of [c]'s b. *)
let b_index c z id kk j =
  let g = c.rg and n = c.rn and k = c.rk in
  if c.rnk then (((((z * g) + id) * n) + j) * k) + kk
  else (((((z * g) + id) * k) + kk) * n) + j

(* A float call: every output of a row in a group has the stated association of
   its products with its group's b, then init; one of a row of no group is init
   alone, or +0, as one of no products. *)
let rows_float_bits c =
  let bk = backend () in
  let name, Stage dt = List.nth stage_dts c.rstage in
  let st = Random.State.make [| c.rseed |] in
  let draw n = drawn_values st ~name ~low:(-8) ~special:16 n in
  let sa, sb, sy = rows_shapes c in
  let count = List.fold_left ( * ) 1 in
  let a, av = stored dt sa (draw (count sa)) in
  let b, bv = stored dt sb (draw (count sb)) in
  let ids = rows_ids st c in
  let outputs = count sy in
  let q = if c.rsplit then 3 else 1 in
  let rows = q * c.rm and n = c.rn and k = c.rk in
  let id_of o = ids.((o / (rows * n) * c.rm) + (o / n mod rows / q)) in
  (* Half the inits of rows of no group are -0, which no addition may touch. *)
  let zeroed o v =
    if (not (in_group c.rg (id_of o))) && Random.State.bool st then -0. else v
  in
  let init =
    if c.rinit then Some (stored dt sy (Array.mapi zeroed (draw outputs)))
    else None
  in
  let dst = Support.on bk (fst (stored dt sy (Array.make outputs nan))) in
  let p = rows_plan c in
  let got, launches =
    rows_call c ~a ~b ~ids ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
  in
  equal answer A.Done got;
  let want =
    Array.init outputs (fun o ->
        let z = o / (rows * n) and i = o / n mod rows and j = o mod n in
        let id = id_of o in
        let init = Option.map (fun (_, v) -> v.(o)) init in
        if in_group c.rg id then
          let id = Int64.to_int id in
          associated_at ~ranges:p.ranges ~k ~init av
            (((z * rows) + i) * k)
            1 bv (b_index c z id 0 j)
            (if c.rnk then 1 else n)
        else associated_at ~ranges:p.ranges ~k:0 ~init av 0 1 bv 0 1)
  in
  let want = A.to_array (A.of_array dt (Iarray.of_list sy) want) in
  equal (array float_exact) want (A.to_array (Support.host dst));
  launches

(* A complex64 call: every output of a row in a group has, in each part, the
   stated association of its real form's 2k terms with its group's b (as
   {!law_complex_bits} states them), then init's part; one of a row of no group
   is init alone, or +0. *)
let rows_complex_bits c =
  let bk = backend () in
  let st = Random.State.make [| c.rseed |] in
  let draw n = drawn_values st ~name:"float32" ~low:(-8) ~special:16 (2 * n) in
  let sa, sb, sy = rows_shapes c in
  let count = List.fold_left ( * ) 1 in
  let a, av = stored_complex sa (draw (count sa)) in
  let b, bv = stored_complex sb (draw (count sb)) in
  let ids = rows_ids st c in
  let outputs = count sy in
  let q = if c.rsplit then 3 else 1 in
  let rows = q * c.rm and n = c.rn and k = c.rk in
  let id_of o = ids.((o / (rows * n) * c.rm) + (o / n mod rows / q)) in
  (* Half the inits of rows of no group are -0, which no addition may touch. *)
  let init =
    if not c.rinit then None
    else
      let vs = draw outputs in
      for o = 0 to outputs - 1 do
        if (not (in_group c.rg (id_of o))) && Random.State.bool st then begin
          vs.(2 * o) <- -0.;
          vs.((2 * o) + 1) <- -0.
        end
      done;
      Some (stored_complex sy vs)
  in
  let dst =
    Support.on bk (fst (stored_complex sy (Array.make (2 * outputs) nan)))
  in
  let p = rows_plan c in
  let got, launches =
    rows_call c ~a ~b ~ids ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
  in
  equal answer A.Done got;
  let part o im =
    let z = o / (rows * n) and i = o / n mod rows and j = o mod n in
    let id = id_of o in
    let init =
      Option.map
        (fun (_, v) -> if im then v.(o).Complex.im else v.(o).Complex.re)
        init
    in
    if not (in_group c.rg id) then
      associated ~ranges:p.ranges ~k:0 ~init (fun _ -> (0., 0.))
    else
      let id = Int64.to_int id in
      associated ~ranges:p.ranges ~k:(2 * k) ~init (fun t ->
          let x = av.((((z * rows) + i) * k) + (t / 2))
          and y = bv.(b_index c z id (t / 2) j) in
          match (t mod 2 = 0, im) with
          | true, false -> (x.Complex.re, y.Complex.re)
          | false, false -> (x.im, -.y.im)
          | true, true -> (x.re, y.im)
          | false, true -> (x.im, y.re))
  in
  let got = A.to_array (Support.host dst) in
  equal ~msg:"real parts" (array float_exact)
    (Array.init outputs (fun o -> part o false))
    (Array.map (fun (x : Complex.t) -> x.re) got);
  equal ~msg:"imaginary parts" (array float_exact)
    (Array.init outputs (fun o -> part o true))
    (Array.map (fun (x : Complex.t) -> x.im) got);
  launches

(* An integer call into int32: every output of a row in a group is the exact sum
   of its products with its group's b, plus init, modulo 2^32; one of a row of
   no group is init alone, or 0. *)
let rows_int_bits c =
  let bk = backend () in
  let st = Random.State.make [| c.rseed |] in
  let acc = D.Any D.Int32 in
  let sa, sb, sy = rows_shapes c in
  let A.Any a, av = int_operand st (D.Any D.Int8) sa ~acc in
  let A.Any b, bv = int_operand st (D.Any D.Int32) sb ~acc in
  let ids = rows_ids st c in
  let count = List.fold_left ( * ) 1 in
  let outputs = count sy in
  let init = if c.rinit then Some (int_operand st acc sy ~acc) else None in
  let dst =
    Support.on bk
      (A.of_array D.Int32 (Iarray.of_list sy) (Array.make outputs (-1l)))
  in
  let got, launches =
    rows_call c ~a ~b ~ids ~dst
      ~init:
        (Option.map
           (fun (A.Any x, _) -> Support.on bk (A.expect D.Int32 (A.Any x)))
           init)
  in
  equal answer A.Done got;
  let q = if c.rsplit then 3 else 1 in
  let rows = q * c.rm and n = c.rn and k = c.rk in
  let sums =
    Array.init outputs (fun o ->
        let z = o / (rows * n) and i = o / n mod rows and j = o mod n in
        let id = ids.((z * c.rm) + (i / q)) in
        let s = ref 0L in
        if in_group c.rg id then begin
          let id = Int64.to_int id in
          for kk = 0 to k - 1 do
            s :=
              Int64.add !s
                (Int64.mul
                   av.((((z * rows) + i) * k) + kk)
                   bv.(b_index c z id kk j))
          done
        end;
        let s =
          match init with None -> !s | Some (_, v) -> Int64.add !s v.(o)
        in
        Int64.to_int32 s)
  in
  equal (array int32) sums (A.to_array (Support.host dst));
  launches

(* [c]'s call, as its kind's law checks it: its launches. *)
let rows_bits c =
  if c.rint then rows_int_bits c
  else if rows_complex c then rows_complex_bits c
  else rows_float_bits c

(* nx_kernel.mli and nx_metal.mli: a Rows call's output multiplies its row's
   group's b alone, with the bits of the stated association, whatever the other
   rows' ids; a row of no group, its id outside [0, g) as an int64, has no
   products: init alone, or +0. At each staged dtype, complex64, and layout of
   b, both grouped tiles, ranges, ids in every arrangement, g = 0, g past m and
   past a chunk of the prepass; integers on the SIMT kernel, exactly. *)
let law_rows_bits c =
  let p = rows_plan c in
  let products = c.rk > 0 && c.rg > 0 in
  cover "float16" (c.rstage = 1 && not c.rint);
  cover "bfloat16" (c.rstage = 2 && not c.rint);
  cover "complex64" (rows_complex c);
  cover "complex64, b n x k" (rows_complex c && c.rnk);
  cover "integers" c.rint;
  cover "32x64" (products && (not c.rint) && p.tile = C.G32);
  cover "16x64" (products && (not c.rint) && p.tile = C.G16);
  cover "m at 16 g" (c.rm = 16 * c.rg && c.rg > 0);
  cover "g past m" (c.rg > c.rm);
  cover "no group" (c.rg = 0);
  cover "one group" (c.rg = 1);
  cover "several chunks of buckets"
    ((c.rbatch * c.rg) + 1 > Nx_metal.Kernels.buckets_chunk);
  cover "ids outside [0, g)" (c.rids = Outside);
  cover "sorted ids" (c.rids = Sorted);
  cover "empty groups" ((c.rids = One || c.rids = Two) && c.rg > 2);
  cover "two ranges" (products && p.ranges = 2);
  cover "more ranges" (products && p.ranges > 2);
  cover "an init" c.rinit;
  cover "a's rows of two axes" c.rsplit;
  cover "b n x k" c.rnk;
  cover "a batch" (c.rbatch > 1);
  cover "no product" (c.rk = 0);
  cover "ids at an offset" (c.rview = Offset_ids);
  cover "stepped ids" (c.rview = Stepped_ids);
  cover "ids broadcast along the batch" (c.rview = Broadcast_ids && c.rbatch > 1);
  cover "ids copied, their batch axes not merging" (c.rview = Split_ids);
  cover "a row of no group with an init of -0"
    (c.rids = Outside && c.rinit && not c.rint);
  cover "rows of no group through the planes"
    (c.rids = Outside && (not c.rinit) && (not c.rint) && products
   && p.ranges > 1);
  let launches = rows_bits c in
  if not c.rint then
    List.iter
      (fun k ->
        cover (Nx_metal.Kernels.name k)
          (List.exists (fun (t : Plan.traced) -> t.kernel = k) launches))
      (List.filter
         (fun k -> grouped_tile k && (not (is_loader k)) && not (banded_twin k))
         (Array.to_list Nx_metal.Kernels.all));
  (* contract.mli: the tiles take the bound's tile rows, the low 32 bits of
     their word of tiles (kernels.h). *)
  List.iter
    (fun (t : Plan.traced) ->
      if grouped_tile t.kernel then
        equal ~msg:"tile rows" int p.tiles_m
          (word t Nx_metal.Kernels.contract_tiles land 0xffff_ffff))
    launches

let rbits_of ?(rstage = 0) ?(rnk = false) ?(rsplit = false) ?(rids = Each)
    ?(rview = Fresh_ids) ?(rinit = false) ?(rint = false) rbatch rm rn rk rg =
  {
    rstage;
    rnk;
    rbatch;
    rm;
    rsplit;
    rn;
    rk;
    rg;
    rids;
    rview;
    rinit;
    rint;
    rseed = 11;
  }

(* Calls that reach each grouped tile at each staged dtype, complex64, and
   layout of b, m at 16 g, g past m, no group, one group, several chunks of
   buckets, ranges, ids at the int64 extremes, integers, each view of the ids,
   no product with and without an init, and rows of no group under an init of -0
   and through the planes. *)
let rbits_examples =
  List.concat_map
    (fun rstage ->
      List.concat_map
        (fun rnk ->
          [
            rbits_of ~rstage ~rnk 1 64 70 40 4;
            rbits_of ~rstage ~rnk 1 20 64 33 8;
          ])
        [ false; true ])
    [ 0; 1; 2; 3 ]
  @ [
      rbits_of ~rids:Outside ~rinit:true 2 48 20 50 3;
      rbits_of ~rids:Sorted 1 128 8 512 384;
      rbits_of 1 20 64 2048 8;
      rbits_of ~rstage:1 ~rinit:true 1 20 64 600 8;
      rbits_of ~rids:Outside ~rinit:true 2 7 9 20 1100;
      rbits_of 1 30 8 10 5000;
      rbits_of ~rinit:true 1 10 10 10 0;
      rbits_of 1 20 16 0 4;
      rbits_of ~rinit:true 2 20 16 0 4;
      rbits_of ~rids:Two ~rsplit:true ~rinit:true 1 40 30 70 12;
      rbits_of ~rint:true ~rids:Outside ~rinit:true 2 20 17 300 6;
      rbits_of ~rint:true ~rnk:true 1 9 5 40 0;
      rbits_of ~rids:Outside 1 20 64 600 8;
      rbits_of ~rview:Offset_ids 1 30 20 40 5;
      rbits_of ~rview:Stepped_ids ~rids:Outside ~rinit:true 2 25 16 50 4;
      rbits_of ~rview:Broadcast_ids ~rnk:true 2 33 20 40 6;
      rbits_of ~rview:Split_ids ~rinit:true 4 20 24 40 3;
      rbits_of ~rint:true ~rview:Split_ids ~rids:Outside 4 10 8 30 3;
      rbits_of 1 20 16 40 1;
    ]
  (* Columns about the grouped tiles' runs and tiles, the last tile at an odd
     column. *)
  @ List.concat_map
      (fun rstage ->
        List.concat_map
          (fun n ->
            [ rbits_of ~rstage 1 64 n 40 4; rbits_of ~rstage 1 20 n 40 8 ])
          [ 7; 8; 15; 16; 63; 64; 65; 129 ])
      [ 0; 1; 2; 3 ]

type rwindowed = { rc : rbits; rallowed : float }

let pp_rwindowed ppf w =
  Format.fprintf ppf "%a, launches of at most %gs" pp_rbits w.rc w.rallowed

(* Calls whose prepass, one launch, fits the least budget: at most 40 groups and
   1024 rows over the batch, a row a prepass thread. Their ids need no copy,
   whose launches of a few elements of several axes take more: split ids' batch
   of 4 becomes 1. *)
let rwindowed_case =
  Gen.(
    with_pp pp_rwindowed
      (let+ rc = rbits_case and+ rallowed = of_list [ 3e-5; 1e-4 ] in
       let split = rc.rview = Split_ids in
       let rview = if split then Fresh_ids else rc.rview in
       let rbatch = if split then 1 else rc.rbatch in
       let q = if rc.rsplit then 3 else 1 in
       let rm = Int.max 1 (Int.min rc.rm (1024 / (rbatch * q))) in
       { rc = { rc with rg = Int.min rc.rg 40; rview; rbatch; rm }; rallowed }))

(* nx_metal.mli: no bit depends on the launches. Under budgets that cut a Rows
   call's grouped tiles or SIMT outputs into windows, or their ranges into
   chunks, every output keeps its bits. *)
let law_rows_window_bits w =
  let module K = Nx_metal.Kernels in
  let launches = bounded w.rallowed (fun () -> rows_bits w.rc) in
  let summing (t : Plan.traced) =
    grouped_tile t.kernel || t.kernel = K.Contract_simt
  in
  let words (t : Plan.traced) =
    if grouped_tile t.kernel then
      (word t K.contract_first, word t K.contract_length)
    else (word t K.simt_first, word t K.simt_length)
  in
  let sums = List.filter summing launches in
  cover "several windows" (List.exists (fun t -> fst (words t) > 0) sums);
  cover "chunks of k" (List.exists (fun t -> snd (words t) > 0) sums)

(* Calls that the least budget cuts into chunks of k, and into several windows,
   on the grouped tiles and on the thread an output. *)
let rwindowed_examples =
  [
    { rc = rbits_of 1 16 64 4096 2; rallowed = 3e-5 };
    { rc = rbits_of ~rstage:1 ~rnk:true 1 16 64 4096 2; rallowed = 3e-5 };
    { rc = rbits_of 2 512 256 64 2; rallowed = 3e-5 };
    { rc = rbits_of ~rint:true 1 16 9 4096 2; rallowed = 3e-5 };
    { rc = rbits_of ~rint:true 2 300 64 64 3; rallowed = 3e-5 };
    { rc = rbits_of 1 16 129 4096 2; rallowed = 3e-5 };
  ]

(* Scaled operands *)

(* How a scale is stored: fresh, its last two axes stepped as a transpose's, its
   first axis broadcast from one element, one element past the start of its
   buffer, or every other element along its last axis of more than one. *)
type sview =
  | Fresh_scale
  | Permuted_scale
  | Broadcast_scale
  | Offset_scale
  | Stepped_scale

(* A scale of an operand: [Exponent] codes or [Value]s of [sdt], and for each
   axis of the operand a ratio drawn as 0 (one), 1 (the whole axis) or 2 (a
   block of 32 where the extent is a multiple of it, else of 2 where it is even,
   else one), or past 2 a block of that many where the extent is a multiple of
   it, else one. *)
type sspec = { sexp : bool; sdt : D.any; sratios : int list; sview : sview }

let pp_sspec ppf s =
  let (D.Any d) = s.sdt in
  Format.fprintf ppf "%s, ratios %s%s"
    (if s.sexp then "Exponent" else "Value of " ^ D.name d)
    (String.concat "," (List.map string_of_int s.sratios))
    (match s.sview with
    | Fresh_scale -> ""
    | Permuted_scale -> ", stored transposed"
    | Broadcast_scale -> ", broadcast"
    | Offset_scale -> ", at an offset"
    | Stepped_scale -> ", stepped")

let ratio_of e = function
  | 0 -> 1
  | 1 -> Int.max 1 e
  | 2 ->
      if e > 0 && e mod 32 = 0 then 32
      else if e > 0 && e mod 2 = 0 then 2
      else 1
  | r -> if e > 0 && e mod r = 0 then r else 1

let sspec_case rank =
  Gen.(
    with_pp pp_sspec
      (let+ sexp = bool
       and+ sdt =
         of_list
           ~pp:(fun ppf (D.Any d) -> D.pp ppf d)
           D.
             [
               Any Float32;
               Any Float16;
               Any Bfloat16;
               Any Float8_e4m3fn;
               Any Int8;
               Any Uint8;
             ]
       and+ sratios = list ~size:(constant rank) (int_range 0 2)
       and+ sview =
         frequency
           [
             (3, constant Fresh_scale);
             (1, constant Permuted_scale);
             (1, constant Broadcast_scale);
             (1, constant Offset_scale);
             (1, constant Stepped_scale);
           ]
       in
       { sexp; sdt; sratios; sview }))

(* Values a [Value] scale draws: powers of two from 2^-4 to 2^4, some times 1.5,
   of either sign, and at times a special. *)
let scale_value st =
  if Random.State.int st 24 = 0 then
    pick_from st [| 0.; -0.; infinity; nan; 0x1p-130 |]
  else
    let v =
      Float.ldexp
        (if Random.State.bool st then 1. else 1.5)
        (Random.State.int st 9 - 4)
    in
    if Random.State.int st 4 = 0 then -.v else v

(* Codes an [Exponent] scale draws: about 127, at times the extremes and 2, 240
   and 241, the edges of the codes whose power of two is a normal float32 that
   keeps every float4_e2m1fn value normal. *)
let scale_code st =
  if Random.State.int st 16 = 0 then
    pick_from st [| 0; 1; 2; 240; 241; 252; 253; 254; 255 |]
  else 120 + Random.State.int st 15

(* nx_metal.mli: an element of float32 value [v] times the code [e]'s 2^(e -
   127), by the arithmetic of its bits: NaN of 255; NaN, an infinity and a zero
   kept; a subnormal [v] a zero of its sign; past the largest float an infinity,
   below 2^-126 a zero of the sign, and the exact product otherwise. *)
let exponent_scaled v e =
  if e = 255 then Float.nan
  else if v = 0. || not (Float.is_finite v) then v
  else if Float.abs v < 0x1p-126 then Float.copy_sign 0. v
  else
    let r = Float.ldexp v (e - 127) in
    if Float.abs r >= 0x1p128 then Float.copy_sign infinity v
    else if Float.abs r < 0x1p-126 then Float.copy_sign 0. v
    else r

(* nx_metal.mli: [v] times a [Value]'s float32 [s], a Mul under the flush:
   subnormal factors read as zeros, a product below 2^-126 a zero of its sign,
   rounded once. *)
let value_scaled v s = to_f32 (flush32 (flush32 v *. flush32 s))

(* The last two of [l] exchanged. *)
let swap_last l =
  match List.rev l with x :: y :: rest -> List.rev (y :: x :: rest) | _ -> l

(* The scale [sp] of an operand of logical [shape], drawn from [st]: the scale
   on [bk], made a view there as [sp] says, and the function that scales the
   float32 value of the operand's element at a logical index. *)
let drawn_scale st bk sp shape =
  let rs = List.map2 ratio_of shape sp.sratios in
  let ss = List.map2 (fun e r -> e / r) shape rs in
  let n = List.fold_left ( * ) 1 ss in
  let rank = List.length ss in
  (* The last axis of more than one element, which a stepped scale steps. *)
  let wide =
    List.fold_left
      (fun w (i, e) -> if e > 1 then Some i else w)
      None
      (List.mapi (fun i e -> (i, e)) ss)
  in
  let view =
    match (sp.sview, wide) with
    | Permuted_scale, _ when rank >= 2 -> `Permuted
    | Broadcast_scale, _ when rank >= 1 && List.hd ss > 0 -> `Broadcast
    | Offset_scale, _ when n > 0 -> `Offset
    | Stepped_scale, Some ax -> `Stepped ax
    | _ -> `Fresh
  in
  (* A broadcast scale's values do not depend on its first axis. *)
  let draw f =
    let v = Array.init n (fun _ -> f st) in
    if view <> `Broadcast then v
    else Array.init n (fun t -> v.(number_of ss (0 :: List.tl (index_of ss t))))
  in
  let stored =
    match view with
    | `Permuted -> swap_last ss
    | `Broadcast -> 1 :: List.tl ss
    | `Fresh | `Offset | `Stepped _ -> ss
  in
  let logical idx = match view with `Permuted -> swap_last idx | _ -> idx in
  let total = List.fold_left ( * ) 1 stored in
  let store v =
    Array.init total (fun t -> v.(number_of ss (logical (index_of stored t))))
  in
  (* The values [v] as [dt] on [bk], laid out as [view] says: [filler] where the
     buffer holds an element the view skips. *)
  let laid dt filler v =
    let xs = store v in
    let make dims vs =
      Contract_cases.on bk
        (Contract_cases.cast_to dt
           (A.Any (A.of_array D.Float64 (Iarray.of_list dims) vs)))
    in
    let moved mv (A.Any x) = A.Any (Option.get (A.move mv x)) in
    let range start count step = { M.start; count; step } in
    match view with
    | `Fresh -> make stored xs
    | `Permuted ->
        let p = swap_last (List.init rank Fun.id) in
        moved (M.Permute (Iarray.of_list p)) (make stored xs)
    | `Broadcast -> moved (M.Broadcast (Iarray.of_list ss)) (make stored xs)
    | `Offset ->
        let x = make [ total + 1 ] (Array.append [| filler |] xs) in
        moved
          (M.Reshape (Iarray.of_list stored))
          (moved (M.Slice [| range 1 total 1 |]) x)
    | `Stepped ax ->
        let doubled =
          List.mapi (fun i e -> if i = ax then 2 * e else e) stored
        in
        let halved idx =
          List.mapi (fun i j -> if i = ax then j / 2 else j) idx
        in
        let at t =
          let idx = index_of doubled t in
          if List.nth idx ax mod 2 = 1 then filler
          else xs.(number_of stored (halved idx))
        in
        let x = make doubled (Array.init (2 * total) at) in
        let step i e = if i = ax then range 0 e 2 else range 0 e 1 in
        let keep = List.mapi step stored in
        moved (M.Slice (Iarray.of_list keep)) x
  in
  let at idx = number_of ss (List.map2 (fun i r -> i / r) idx rs) in
  if sp.sexp then
    let codes = draw scale_code in
    let e = laid (D.Any D.Uint8) 255. (Array.map float_of_int codes) in
    ( Nx_kernel.Exponent (A.expect D.Uint8 e),
      fun idx v -> exponent_scaled v codes.(at idx) )
  else
    let vals = draw scale_value in
    let held =
      A.to_array
        (A.expect D.Float32
           (Contract_cases.cast_to (D.Any D.Float32)
              (Contract_cases.cast_to sp.sdt
                 (A.Any (A.of_array D.Float64 (Iarray.of_list ss) vals)))))
    in
    let (A.Any v) = laid sp.sdt 7. vals in
    (Nx_kernel.Value v, fun idx e -> value_scaled e held.(at idx))

(* The layout of scale [s] on the device. *)
let scale_layout : Nx_kernel.scale -> A.Layout.t = function
  | Exponent e -> A.layout e
  | Value v -> A.layout v

(* Whether layout [l] starts past its buffer's first element, and whether it
   skips elements between its first and its last. *)
let past_start l = fst (A.Layout.span l) > 0

let skips l =
  let lo, hi = A.Layout.span l in
  let reached = ref 1 in
  Iarray.iteri
    (fun i e -> if A.Layout.stride l i <> 0 then reached := !reached * e)
    (A.Layout.shape l);
  hi - lo > !reached

(* nx_metal.mli: the loaders' keys, the weights the models store: b's dtype,
   whether its scale is of Exponent codes and a Value scale's dtype, each by a
   of float32, float16 or bfloat16. *)
let loader_keys =
  let rows = D.[ Any Float32; Any Float16; Any Bfloat16 ] in
  D.
    [
      (Any Float4_e2m1fn, true, Any Uint8, rows);
      (Any Int8, false, Any Float16, rows);
      (Any Int8, false, Any Float32, rows);
    ]

(* contract.mli: a loader's scale steps lie within signed 32 bits, a block's
   elements too, and b's terms number under 2^32; one past, the call reads b's
   products from workspace. Of b [1; k; n] k x n and a scale of [1; k / rk; n]:
   each row its scale's stride along the columns [sn] at its bound and past,
   along the terms of [rk] elements a block, and [k] terms of blocks of one. *)
let step_rows =
  let top = (1 lsl 31) - 1 in
  [
    (`Sn top, true);
    (`Sn (top + 1), false);
    (`Sn (-top - 1), true);
    (`Sn (-top - 2), false);
    (`Rk top, true);
    (`Rk (top + 1), false);
    (`K ((1 lsl 32) - 1), true);
    (`K (1 lsl 32), false);
  ]

(* kernels.h: a loader's divisor word of d, a multiplier m in its low 32 bits
   and shifts s1 and s2 in bits 32-39 and 40-47, gives u / d as (t + ((u - t)
   lsr s1)) lsr s2, t the high 32 bits of u m, for every u below 2^32 and d from
   1 to 2^32 - 1. *)
let divided w u =
  let m = w land 0xFFFF_FFFF
  and s1 = (w lsr 32) land 255
  and s2 = (w lsr 40) land 255 in
  let t = Int64.(to_int (shift_right_logical (mul (of_int u) (of_int m)) 32)) in
  (t + ((u - t) lsr s1)) lsr s2

let top32 = (1 lsl 32) - 1

(* Divisors at powers of two and their neighbours, small primes, and the
   domain's ends; numerators at 0, about d and 2 d, and 2^32's top. *)
let divisor_edges =
  [
    1;
    2;
    3;
    5;
    7;
    31;
    32;
    33;
    641;
    65535;
    65536;
    65537;
    1 lsl 30;
    (1 lsl 31) - 1;
    1 lsl 31;
    (1 lsl 31) + 1;
    top32 - 1;
    top32;
  ]

let numerators d =
  List.filter
    (fun u -> u >= 0 && u <= top32)
    [ 0; 1; d - 1; d; d + 1; (2 * d) - 1; 2 * d; top32 - d; top32 ]

type division = { d : int; u : int }

let pp_division ppf c = Format.fprintf ppf "%d / %d" c.u c.d

let division_examples =
  List.concat_map
    (fun d -> List.map (fun u -> { d; u }) (numerators d))
    divisor_edges

let division =
  let open Gen in
  with_pp pp_division
    (let* d =
       frequency
         [
           (1, of_list divisor_edges);
           (1, map (fun e -> 1 lsl e) (int_range 0 31));
           (2, int_range 1 top32);
         ]
     in
     let+ u =
       frequency [ (1, of_list (numerators d)); (2, int_range 0 top32) ]
     in
     { d; u })

let law_divisor c =
  cover "a power of two" (c.d land (c.d - 1) = 0);
  cover "d above 2^31" (c.d > 1 lsl 31);
  cover "u at 2^32's top" (c.u = top32);
  equal int (c.u / c.d) (divided (C.divisor c.d) c.u)

(* contract.mli: a divisor word of d from 1 to 2^32 - 1 only. *)
let test_divisor_domain () =
  raises (Invalid_argument "Contract.divisor") (fun () -> C.divisor 0);
  raises (Invalid_argument "Contract.divisor") (fun () -> C.divisor (1 lsl 32));
  raises (Invalid_argument "Contract.divisor") (fun () -> C.divisor (-1))

let step_row_id = function
  | `Sn s, _ -> Printf.sprintf "a column stride of %d" s
  | `Rk r, _ -> Printf.sprintf "blocks of %d terms" r
  | `K k, _ -> Printf.sprintf "%d terms" k

let test_step_row (row, fits) =
  let module L = A.Layout in
  let lay ?(offset = 0) shape strides =
    L.v ~offset ~strides:(Iarray.of_list strides) (Iarray.of_list shape)
  in
  let l, sl =
    match row with
    | `Sn s ->
        let offset = if s < 0 then -s else 0 in
        (lay [ 1; 2; 2 ] [ 4; 2; 1 ], lay ~offset [ 1; 2; 2 ] [ 0; 1; s ])
    | `Rk r -> (lay [ 1; r; 2 ] [ 0; 2; 1 ], lay [ 1; 1; 2 ] [ 0; 2; 1 ])
    | `K k -> (lay [ 1; k; 2 ] [ 0; 2; 1 ], lay [ 1; k; 2 ] [ 0; 2; 1 ])
  in
  let st = C.scale_steps ~nk:false l sl [| [| 0 |]; [| 1 |]; [| 2 |] |] in
  equal bool fits (Option.is_some st);
  match (row, st) with
  | `Sn s, Some st -> equal int s st.sn
  | `Rk r, Some st -> equal int r st.rk
  | `K _, Some st -> equal int 1 st.rk
  | _, None -> ()

(* nx_metal.mli: whether a call of an unscaled a of [a] and a b of [b] scaled by
   [s], of [shape] laid out as [l], takes a loader at [tile]: a loader's key, at
   a tile other than 64 x 64, b's scale of one block or of blocks of one element
   along each of b's [lead] axes before its terms and columns, and a b narrower
   than a byte with no stride negative and its columns or its terms, the axis
   [terms] of [shape], contiguous. *)
let loader_key ~a ~b = function
  | None -> false
  | Some s ->
      List.exists
        (fun (kb, exp, sdt, ka) ->
          kb = b && exp = s.sexp && (exp || s.sdt = sdt) && List.mem a ka)
        loader_keys

(* nx_metal.mli: the dtype a scaled call's tiles' rows of [d], scaled by [s],
   enter a loader at: their cast into float32, unscaled and of no loader's rows'
   dtype. *)
let entered d (s : sspec option) =
  if Option.is_none s && not (List.mem d native) then D.Any D.Float32 else d

let loads ~a ~b ~lead ~tile ~terms shape l = function
  | None -> false
  | Some s ->
      let fits i =
        let e = List.nth shape i in
        let r = ratio_of e (List.nth s.sratios i) in
        r = 1 || e / r <= 1
      in
      let rank = List.length shape in
      let strides = List.init rank (A.Layout.stride l) in
      let contiguous i = List.nth strides i = 1 || List.nth shape i <= 1 in
      let cols = if terms = rank - 1 then rank - 2 else rank - 1 in
      let in_place =
        let (D.Any d) = b in
        D.bits d >= 8
        || List.for_all (fun st -> st >= 0) strides
           && (contiguous cols || contiguous terms)
      in
      loader_key ~a ~b (Some s) && tile <> C.T64
      && List.for_all fits (List.init lead Fun.id)
      && in_place

(* The launches of the pass that writes a scaled operand's products. *)
let scale_passes launches =
  List.filter_map
    (fun (t : Plan.traced) ->
      if t.kernel = Nx_metal.Kernels.Contract_scale then
        Some (Nx_metal.Kernels.name t.kernel)
      else None)
    launches

type scbits = { sc : cvbits; ssa : sspec option; ssb : sspec option }

let pp_scbits ppf w =
  let pp_scaled what ppf = function
    | None -> ()
    | Some s -> Format.fprintf ppf "; %s scaled by %a" what pp_sspec s
  in
  Format.fprintf ppf "%a%a%a" pp_cvbits w.sc (pp_scaled "a") w.ssa
    (pp_scaled "b") w.ssb

(* Scaled calls: a's, b's or both scales, at times both operands float32, and at
   times b 2, 4, 6 or 8 elements into its buffer: an e2m1 b 1, 2, 3 or 4 bytes
   in. *)
let scbits_case =
  Gen.(
    with_pp pp_scbits
      (let+ sc = cvbits_case
       and+ f32 = frequency [ (2, constant false); (1, constant true) ]
       and+ which = int_range 1 3
       and+ sa = sspec_case 3
       and+ sb = sspec_case 3
       and+ boff =
         frequency
           [ (5, constant None); (1, map Option.some (of_list [ 2; 4; 6; 8 ])) ]
       in
       let sc =
         if f32 then { sc with va = D.Any D.Float32; vb = D.Any D.Float32 }
         else sc
       in
       let sc =
         match boff with Some s -> { sc with vview = Offset_b s } | None -> sc
       in
       {
         sc;
         ssa = (if which land 1 <> 0 then Some sa else None);
         ssb = (if which land 2 <> 0 then Some sb else None);
       }))

let tile_prefix prefix k = String.starts_with ~prefix (Nx_metal.Kernels.name k)

(* nx_kernel.mli and nx_metal.mli: a scaled operand's element is its cast into
   float32 times its scale block's scale, rounded once, by the rule of its kind,
   and the call stages at float32: every output has the bits of the stated
   association of the scaled values' products, the float32 plan's, then init,
   rounded once into dst. *)
let law_scaled_bits w =
  let c = w.sc in
  let bk = backend () in
  let st = Random.State.make [| c.vseed |] in
  let bt = c.vbatch and m = c.vm and n = c.vn and k = c.vk in
  let low = c.vlow in
  let a, raw_a, b, raw_b = cv_operands st bk c in
  let a_index z i kk = [ z; i; kk ] in
  let b_index z kk j = if c.vnk then [ z; j; kk ] else [ z; kk; j ] in
  let sa = Option.map (fun sp -> drawn_scale st bk sp [ bt; m; k ]) w.ssa in
  let sb =
    Option.map
      (fun sp ->
        drawn_scale st bk sp (if c.vnk then [ bt; n; k ] else [ bt; k; n ]))
      w.ssb
  in
  let av z i kk =
    match sa with
    | None -> raw_a z i kk
    | Some (_, f) -> f (a_index z i kk) (raw_a z i kk)
  in
  let bv z kk j =
    match sb with
    | None -> raw_b z kk j
    | Some (_, f) -> f (b_index z kk j) (raw_b z kk j)
  in
  let shape = [ bt; m; n ] in
  let outputs = bt * m * n in
  let (D.Any ddt) = c.vdst in
  let init =
    if not c.vinit then None
    else
      let name = D.name ddt in
      let vs = drawn_values st ~name ~low ~special:16 outputs in
      let made = A.of_array D.Float64 (Iarray.of_list shape) vs in
      let x = A.expect ddt (Contract_cases.cast_to c.vdst (A.Any made)) in
      let held = Contract_cases.cast_to (D.Any D.Float32) (A.Any x) in
      Some (x, A.to_array (A.expect D.Float32 held))
  in
  let dst = A.create (S.device ()) ddt (Iarray.of_list shape) in
  let p =
    planned
      {
        C.batch = bt;
        m;
        n;
        k;
        stage = C.F32;
        nk = c.vnk;
        complex = false;
        source = 0;
      }
  in
  let specs = List.filter_map Fun.id [ w.ssa; w.ssb ] in
  let narrow d =
    let (D.Any d) = d in
    D.bits d < 8
  in
  let along_k sp kaxis =
    List.nth sp.sratios kaxis = 2 && k mod 32 = 0 && k > 0
  in
  cover "Exponent" (List.exists (fun s -> s.sexp) specs);
  cover "Value" (List.exists (fun s -> not s.sexp) specs);
  cover "a scaled" (Option.is_some w.ssa && Option.is_none w.ssb);
  cover "b scaled" (Option.is_none w.ssa && Option.is_some w.ssb);
  cover "both scaled" (Option.is_some w.ssa && Option.is_some w.ssb);
  cover "blocks of 32 along k"
    (Option.fold ~none:false ~some:(fun s -> along_k s 2) w.ssa
    || Option.fold ~none:false
         ~some:(fun s -> along_k s (if c.vnk then 2 else 1))
         w.ssb);
  cover "a scale stored transposed"
    (List.exists (fun s -> s.sview = Permuted_scale) specs);
  cover "a broadcast scale"
    (List.exists (fun s -> s.sview = Broadcast_scale) specs);
  cover "a scaled operand narrower than a byte"
    ((Option.is_some w.ssa && narrow c.va)
    || (Option.is_some w.ssb && narrow c.vb));
  let scales = List.filter_map (Option.map fst) [ sa; sb ] in
  cover "a scale at an offset"
    (List.exists (fun s -> past_start (scale_layout s)) scales);
  cover "a stepped scale" (List.exists (fun s -> skips (scale_layout s)) scales);
  cover "a scaled operand narrower than a byte at a nonzero bit phase"
    ((phased a && Option.is_some sa) || (phased b && Option.is_some sb));
  cover "a strided a" c.vat;
  cover "b n x k" c.vnk;
  cover "exchanged" p.exchanged;
  cover "several ranges" (k > 0 && p.ranges > 1);
  cover "an init" c.vinit;
  let operand x s =
    { Nx_kernel.array = Contract_cases.on bk x; scale = Option.map fst s }
  in
  let oa = operand a sa and ob = operand b sb in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (2, if c.vnk then 2 else 1) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
      oa ob
  in
  equal answer A.Done got;
  let ran prefix =
    List.exists (fun (t : Plan.traced) -> tile_prefix prefix t.kernel) launches
  in
  cover "the float32 tiles" (ran "contract_f32_");
  (* The tiles' rows and columns: a and b, or b and a where exchanged. *)
  let keyed =
    let a_dt, rows_s, b_dt, cols_s, (cols : Nx_kernel.operand), shape =
      if p.exchanged then (c.vb, w.ssb, c.va, w.ssa, oa, [ bt; m; k ])
      else
        ( c.va,
          w.ssa,
          c.vb,
          w.ssb,
          ob,
          if c.vnk then [ bt; n; k ] else [ bt; k; n ] )
    in
    let (A.Any x) = cols.array in
    let a_dt = entered a_dt rows_s in
    let key = Option.is_none rows_s && loader_key ~a:a_dt ~b:b_dt cols_s in
    let l = A.layout x in
    cover "a key's call at 64 x 64" (key && p.tile = C.T64);
    cover "a key's scale of several blocks along the batch"
      (key && bt > 1
      && Option.fold ~none:false
           ~some:(fun s -> List.nth s.sratios 0 = 2 && bt / ratio_of bt 2 > 1)
           cols_s);
    cover "a key's b narrower than a byte with a negative stride"
      (key
      && b_dt = D.Any D.Float4_e2m1fn
      && List.exists (fun i -> A.Layout.stride l i < 0) [ 0; 1; 2 ]);
    let terms = if p.exchanged || c.vnk then 2 else 1 in
    let taken =
      key && loads ~a:a_dt ~b:b_dt ~lead:1 ~tile:p.tile ~terms shape l cols_s
    in
    cover "a loader's b narrower than a byte of its columns contiguous alone"
      (taken && narrow b_dt
      && A.Layout.stride l terms <> 1
      && List.nth shape terms > 1);
    cover "a loader's b narrower than a byte at a nonzero bit phase"
      (taken && phased cols.array);
    cover "a loader's b of bytes at an odd offset"
      (taken && b_dt = D.Any D.Int8 && A.Layout.offset l mod 2 = 1);
    (* A loader tile of b n x k stages a thread's run of [run] terms as one
       where its scale blocks along k are of a multiple of [run] terms and at
       least a step's [depth], else element by element. A run's scale block
       wraps to a nonzero term where its blocks are not of whole steps. *)
    let nk_form = p.exchanged || c.vnk in
    let depth = C.depth C.F32 p.tile in
    let run = depth * C.cols p.tile / C.threads in
    let rk =
      Option.fold ~none:0
        ~some:(fun s -> ratio_of k (List.nth s.sratios (List.length shape - 1)))
        cols_s
    in
    let runs = taken && nk_form && rk >= depth && rk mod run = 0 in
    cover "a loader tile staging runs" runs;
    cover "a scaled n x k loader tile staging element by element"
      (taken && nk_form && not runs);
    cover "a loader's runs past a scale block's start"
      (runs && rk mod depth <> 0);
    (* An e2m1 b's runs of whole 32-bit words, read a word at a time where every
       run starts on 4 bytes: index 0 on 4 bytes of its fresh buffer and its
       batch's and columns' steps whole words. Such runs need scale blocks of a
       multiple of 16 terms that divide k, so a row is whole words. With no
       tail, the last run ends at the buffer's last byte. *)
    let e2m1_runs =
      runs && b_dt = D.Any D.Float4_e2m1fn && run * 4 mod 32 = 0
    in
    let on_words = 4 * A.Layout.offset l mod 32 = 0 in
    let steps_words =
      List.for_all
        (fun i -> 4 * A.Layout.stride l i mod 32 = 0)
        [ 0; List.length shape - 2 ]
    in
    let byte_in = 4 * A.Layout.offset l mod 32 / 8 in
    cover "an e2m1 loader's runs of words, read as words"
      (e2m1_runs && on_words && steps_words);
    cover "an e2m1 loader's runs of words 1, 2 or 3 bytes off 4"
      (e2m1_runs && List.mem byte_in [ 1; 2; 3 ]);
    cover "an e2m1 loader's runs read as words to the buffer's last byte"
      (e2m1_runs && on_words && steps_words
      && k mod depth = 0
      && (c.vview = Fresh_operands
         || match c.vview with Offset_b _ -> true | _ -> false));
    (* A k x n e2m1 b's runs, of adjacent columns at one term, read as words
       where they start on 4 bytes as above, else by bytes. *)
    let kn_runs =
      taken && (not nk_form)
      && b_dt = D.Any D.Float4_e2m1fn
      && run * 4 mod 32 = 0
    in
    cover "an e2m1 loader's k x n runs, read as words"
      (kn_runs && on_words && steps_words);
    cover "an e2m1 loader's k x n runs 1, 2 or 3 bytes off 4"
      (kn_runs && List.mem byte_in [ 1; 2; 3 ]);
    taken
  in
  check_loaders ~keyed launches;
  cover_kernels (loaders ~rows:false ~scaled:true) launches;
  (* nx_metal.mli: a loader reads a scaled b and its scales where they lie,
     without the workspace of its products. *)
  if keyed then equal (list string) [] (scale_passes launches);
  let _, _, sums, sum, _ =
    reference ~ranges:p.ranges ~bt ~m ~n ~k ~init:(Option.map snd init) av bv
  in
  let held = A.Any (A.of_array D.Float32 (Iarray.of_list shape) sums) in
  let as64 x =
    A.to_array (A.expect D.Float64 (Contract_cases.cast_to (D.Any D.Float64) x))
  in
  let want = as64 (Contract_cases.cast_to c.vdst held) in
  (* Output [o] with term [kk] made +0, as dst holds it. *)
  let dropped o kk =
    let v = sum ~drop:kk o in
    (as64
       (Contract_cases.cast_to c.vdst
          (A.Any (A.of_array D.Float32 [| 1 |] [| v |])))).(0)
  in
  cover_outputs ~da:c.va ~db:c.vb ~ranges:p.ranges ~outputs
    ~want:(fun o -> want.(o))
    ~dropped
    ~init:(fun o -> Option.fold ~none:0. ~some:(fun (_, v) -> v.(o)) init)
    ~terms:(fun o ->
      let z = o / (m * n) and i = o / n mod m and j = o mod n in
      Array.init k (fun kk -> (raw_a z i kk, raw_b z kk j)));
  equal (array float_exact) want (as64 (A.Any (Support.host dst)))

let sspec_of ?(sexp = true) ?(sdt = D.Any D.Float32) ?(sview = Fresh_scale)
    sratios =
  { sexp; sdt; sratios; sview }

let scbits_of ?ssa ?ssb sc = { sc; ssa; ssb }

(* [f ~nk ~s a b] for each loader's key, at each layout of b, [s] b's scale of
   blocks of 32 along k, b of [rank] axes, k and n last. *)
let scaled_loader_examples ~rank f =
  let along ~nk =
    List.init rank (fun i -> if i = rank - if nk then 1 else 2 then 2 else 0)
  in
  List.concat_map
    (fun (b, sexp, sdt, ka) ->
      List.concat_map
        (fun a ->
          List.concat_map
            (fun nk -> f ~nk ~s:(sspec_of ~sexp ~sdt (along ~nk)) a b)
            [ false; true ])
        ka)
    loader_keys

(* Calls that reach the float32 tiles and the casts, each kind of scale on a, on
   b and on both, blocks of 32 along k, a whole axis, a transposed, a broadcast,
   an offset and a stepped scale, a scaled operand narrower than a byte, inside
   a byte too, and float8 codes reaching finite, NaN and infinite outputs over
   several ranges. *)
let scbits_examples =
  let open D in
  let f32 = Any Float32 in
  let blocks = sspec_of [ 0; 0; 2 ]
  and rows = sspec_of ~sexp:false [ 0; 0; 1 ] in
  [
    scbits_of ~ssa:blocks (cvbits_of f32 f32 1 64 64 64);
    scbits_of
      ~ssb:(sspec_of [ 0; 2; 0 ])
      (cvbits_of (Any Float8_e4m3fn) (Any Float16) 1 64 64 64);
    scbits_of
      ~ssb:(sspec_of [ 0; 0; 2 ])
      (cvbits_of ~vnk:true f32 (Any Float8_e4m3fn) 2 40 64 96);
    scbits_of ~ssa:rows
      ~ssb:
        (sspec_of ~sexp:false ~sdt:(Any Bfloat16) ~sview:Permuted_scale
           [ 0; 2; 0 ])
      (cvbits_of f32 (Any Int8) 1 30 50 64);
    scbits_of
      ~ssa:(sspec_of ~sview:Broadcast_scale [ 0; 1; 2 ])
      (cvbits_of ~vat:true (Any Int4) f32 3 20 20 64);
    (* An exchanged call whose weight, a of float4_e2m1fn, has its columns
       contiguous and its terms 17 apart: a loader reads it k x n. *)
    scbits_of
      ~ssa:(sspec_of [ 0; 0; 0 ])
      (cvbits_of ~vat:true (Any Float4_e2m1fn) f32 1 17 1 2);
    scbits_of
      ~ssb:(sspec_of ~sexp:false ~sdt:(Any Float16) [ 1; 1; 1 ])
      (cvbits_of ~vinit:true f32 f32 1 100 128 40);
    scbits_of ~ssa:blocks (cvbits_of (Any Float16) (Any Int8) 1 64 16 40);
    scbits_of
      ~ssb:(sspec_of [ 0; 2; 0 ])
      (cvbits_of f32 (Any Float8_e4m3fn) 1 20 20 2048);
    scbits_of
      ~ssa:(sspec_of ~sview:Offset_scale [ 0; 0; 2 ])
      ~ssb:
        (sspec_of ~sexp:false ~sdt:(Any Float16) ~sview:Stepped_scale
           [ 0; 2; 0 ])
      (cvbits_of ~vview:Offset_operands (Any Int4) (Any Float4_e2m1fn) 1 30 40
         64);
    scbits_of
      ~ssb:(sspec_of ~sexp:false ~sview:Offset_scale [ 1; 0; 0 ])
      (cvbits_of ~vat:true ~vview:Offset_operands f32 (Any Bit) 2 20 30 50);
    scbits_of
      ~ssa:(sspec_of [ 0; 0; 2 ])
      (cvbits_of (Any Float8_e5m2) f32 1 64 64 2048);
    scbits_of
      ~ssb:(sspec_of [ 0; 2; 0 ])
      (cvbits_of (Any Float8_e5m2) f32 1 160 64 64);
  ]
  @ [
      (* Keys' calls that take no loader: at 64 x 64, of a scale of several
         blocks along the batch, of a float4 b flipped. *)
      scbits_of
        ~ssb:(sspec_of ~sexp:false ~sdt:(Any Float16) [ 0; 2; 0 ])
        (cvbits_of f32 (Any Int8) 4 320 320 32);
      scbits_of
        ~ssb:(sspec_of ~sexp:false ~sdt:(Any Float16) [ 2; 2; 0 ])
        (cvbits_of f32 (Any Int8) 4 16 64 64);
      scbits_of
        ~ssb:(sspec_of [ 0; 2; 0 ])
        (cvbits_of ~vview:Flipped (Any Bfloat16) (Any Float4_e2m1fn) 1 16 64 64);
      (* Keys' b of terms contiguous in scale blocks of 36 and 48 terms, which a
         tile's steps of 16 or 32 cross; of a scale an element. *)
      scbits_of
        ~ssb:(sspec_of [ 0; 0; 36 ])
        (cvbits_of ~vnk:true (Any Bfloat16) (Any Float4_e2m1fn) 1 40 64 144);
      scbits_of
        ~ssb:(sspec_of [ 0; 0; 48 ])
        (cvbits_of ~vnk:true (Any Bfloat16) (Any Float4_e2m1fn) 1 40 64 144);
      scbits_of
        ~ssb:(sspec_of ~sexp:false ~sdt:(Any Float16) [ 0; 0; 0 ])
        (cvbits_of ~vnk:true (Any Float32) (Any Int8) 1 40 64 64);
      (* Keys' b of terms contiguous an odd element in: an e2m1 run at bit phase
         4, an int8 run at an odd byte. *)
      scbits_of
        ~ssb:(sspec_of [ 0; 0; 2 ])
        (cvbits_of ~vnk:true ~vview:Offset_operands (Any Bfloat16)
           (Any Float4_e2m1fn) 1 40 64 64);
      scbits_of
        ~ssb:(sspec_of ~sexp:false ~sdt:(Any Float16) [ 0; 0; 2 ])
        (cvbits_of ~vnk:true ~vview:Offset_operands (Any Float32) (Any Int8) 1
           40 64 64);
    ]
  (* Keys' e2m1 b of terms contiguous in runs of whole words: 1, 2, 3 and 4
     bytes into its buffer, the last run at its last byte; of columns
     contiguous, fresh, 1 and 4 bytes in; rows of 68 terms, which end inside a
     word, of runs of 4 terms. *)
  @ List.map
      (fun s ->
        scbits_of
          ~ssb:(sspec_of [ 0; 0; 2 ])
          (cvbits_of ~vnk:true ~vview:(Offset_b s) (Any Bfloat16)
             (Any Float4_e2m1fn) 1 16 64 64))
      [ 2; 4; 6; 8 ]
  @ List.map
      (fun v ->
        scbits_of
          ~ssb:(sspec_of [ 0; 2; 0 ])
          (cvbits_of ~vview:v (Any Bfloat16) (Any Float4_e2m1fn) 2 16 96 64))
      [ Fresh_operands; Offset_b 2; Offset_b 8 ]
  @ [
      scbits_of
        ~ssb:(sspec_of [ 0; 0; 32 ])
        (cvbits_of ~vnk:true (Any Bfloat16) (Any Float4_e2m1fn) 1 16 64 68);
      scbits_of
        ~ssb:(sspec_of [ 0; 0; 32 ])
        (cvbits_of ~vnk:true ~vview:(Offset_b 8) (Any Float32)
           (Any Float4_e2m1fn) 2 16 64 128);
    ]
  @ scaled_loader_examples ~rank:3 (fun ~nk ~s a b ->
      [
        scbits_of ~ssb:s (cvbits_of ~vnk:nk a b 1 16 64 64);
        scbits_of ~ssb:s (cvbits_of ~vnk:nk a b 1 40 64 64);
      ])

(* nx_metal.mli: the dtype of a float32 call's stage, float32 where the call is
   [scaled]. *)
let stage_dtype ~scaled da db =
  if scaled then D.Any D.Float32
  else
    match converted_stage da db with
    | C.F16 -> D.Any D.Float16
    | C.Bf16 -> D.Any D.Bfloat16
    | C.F32 | C.I8 -> D.Any D.Float32

(* nx_metal.mli: a vector key's bounds on an unsegmented call's rows and terms,
   of an operand of [d] scaled by [s], or 0 rows for no key. *)
let key_bounds d (s : Nx_kernel.scale option) =
  let values (A.Any v) =
    List.mem (D.Any (A.dtype v)) D.[ Any Float16; Any Float32 ]
  in
  match (d, s) with
  | D.Any D.Float32, None -> (1, 1 lsl 23)
  | (D.Any D.Float16 | D.Any D.Bfloat16), None -> (2, 1 lsl 22)
  | D.Any D.Float4_e2m1fn, Some (Exponent _) -> (11, 1 lsl 19)
  | D.Any D.Int8, Some (Value v) when values (A.Any v) -> (3, 1 lsl 19)
  | _ -> (0, 0)

(* nx_metal.mli: whether an unsegmented float32 call of [m] rows, [n] columns
   and [k] terms of a of [da] scaled by [sa] and b of [db] scaled by [sb] takes
   the vector kernels on its own dtypes. *)
let vector_own ~m ~n ~k da sa db sb =
  let other d s = Option.is_some s || List.mem d native in
  let bb, tb = key_bounds db sb and ba, ta = key_bounds da sa in
  k > 0
  && ((bb > 0 && other da sa && m <= bb && n >= 256 && k <= tb)
     || (ba > 0 && other db sb && n <= ba && m >= 256 && k <= ta))

(* nx_metal.mli: whether a loader takes a scaled float32 call on its operands'
   own dtypes: its tiles' rows ([b] and [a] where exchanged) unscaled, of
   float32, float16 or bfloat16, and its columns a weight it reads in place, at
   a tile but 64 x 64. *)
let loader_own ~bt ~m ~n ~k ~nk (a, da, sa) (b, db, sb) =
  let p =
    planned
      { C.batch = bt; m; n; k; stage = C.F32; nk; complex = false; source = 0 }
  in
  let rows_dt, rows_s, cols, cols_dt, cols_s, shape =
    if p.exchanged then (db, sb, a, da, sa, [ bt; m; k ])
    else (da, sa, b, db, sb, if nk then [ bt; n; k ] else [ bt; k; n ])
  in
  let (A.Any x) = cols in
  k > 0 && Option.is_none rows_s
  && loads ~a:rows_dt ~b:cols_dt ~lead:1 ~tile:p.tile
       ~terms:(if p.exchanged || nk then 2 else 1)
       shape (A.layout x) cols_s

(* nx_metal.mli: a float32 call of an unscaled operand not of its stage's dtype,
   which the vector kernels do not take on its own dtypes, is the call of that
   operand's cast into it: every output has the bits of the call of the operands
   cast by nx.cpu, C order on the device, beside the same scales, init and dst,
   at every layout, on the tiles, the loaders and the vector kernels. *)
let law_mixed_casts w =
  let c = w.sc in
  let bk = backend () in
  let st = Random.State.make [| c.vseed |] in
  let bt = c.vbatch and m = c.vm and n = c.vn and k = c.vk in
  let a, _, b, _ = cv_operands st bk c in
  let b_shape = if c.vnk then [ bt; n; k ] else [ bt; k; n ] in
  let scale sp shape = fst (drawn_scale st bk sp shape) in
  let sa = Option.map (fun sp -> scale sp [ bt; m; k ]) w.ssa in
  let sb = Option.map (fun sp -> scale sp b_shape) w.ssb in
  let scaled = Option.is_some sa || Option.is_some sb in
  let stage = stage_dtype ~scaled c.va c.vb in
  let recast d s = Option.is_none s && d <> stage in
  let cast x d s =
    if recast d s then
      let host = Contract_cases.host x in
      Contract_cases.on bk (Contract_cases.cast_to stage host)
    else x
  in
  let shape = [ bt; m; n ] in
  let (D.Any ddt) = c.vdst in
  let init =
    if not c.vinit then None
    else
      let name = D.name ddt and outputs = bt * m * n in
      let vs = drawn_values st ~name ~low:c.vlow ~special:16 outputs in
      let made = A.of_array D.Float64 (Iarray.of_list shape) vs in
      let x = Contract_cases.cast_to c.vdst (A.Any made) in
      Some (Support.on bk (A.expect ddt x))
  in
  let run a sa b sb =
    let dst = A.create (S.device ()) ddt (Iarray.of_list shape) in
    let got, launches =
      Plan.traced @@ fun () ->
      Nx_metal.contract
        ~batch:(Iarray.of_list [ (0, 0) ])
        ~contracting:(Iarray.of_list [ (2, if c.vnk then 2 else 1) ])
        ~segments:None ~acc:(D.Any D.Float32) ~dst ~init
        { Nx_kernel.array = a; scale = sa }
        { Nx_kernel.array = b; scale = sb }
    in
    equal answer A.Done got;
    (Support.elements (A.Any dst), launches)
  in
  let got, launches = run a sa b sb in
  let own = vector_own ~m ~n ~k c.va sa c.vb sb in
  let cast x d s = if own then x else cast x d s in
  let want, cast_launches = run (cast a c.va sa) sa (cast b c.vb sb) sb in
  let ran l =
    List.exists (fun (t : Plan.traced) -> tile_prefix "contract_mv_" t.kernel) l
  in
  let cast_a = (not own) && recast c.va sa in
  let cast_b = (not own) && recast c.vb sb in
  cover "a cast at float16" (stage = D.Any D.Float16 && (cast_a || cast_b));
  cover "a cast at bfloat16" (stage = D.Any D.Bfloat16 && (cast_a || cast_b));
  cover "a cast at float32" (stage = D.Any D.Float32 && (cast_a || cast_b));
  cover "both operands cast" (cast_a && cast_b);
  cover "a scaled call's unscaled operand cast" (scaled && (cast_a || cast_b));
  cover "the vector kernels, a cast" (ran launches && (cast_a || cast_b));
  cover "a weight broadcast along a batch of 64, cast"
    (c.vview = Broadcast_b && bt = 64 && cast_b);
  cover "a flipped operand, cast" (c.vview = Flipped && (cast_a || cast_b));
  cover "a strided a, cast" (c.vat && cast_a);
  let tile =
    let stage =
      match stage with
      | D.Any D.Float16 -> C.F16
      | D.Any D.Bfloat16 -> C.Bf16
      | _ -> C.F32
    in
    let e =
      { C.batch = bt; m; n; k; stage; nk = c.vnk; complex = false; source = 0 }
    in
    Option.map (fun (p : C.t) -> p.tile) (C.plan e)
  in
  let tiled t = (cast_a || cast_b) && (not (ran launches)) && tile = Some t in
  cover "64 x 64 tiles, a cast" (tiled C.T64);
  cover "32 x 32 tiles, a cast" (tiled C.T32);
  cover "16 x 64 tiles, a cast" (tiled C.T16);
  let nan_code (A.Any x) =
    match A.dtype x with
    | D.Float8_e4m3fn | D.Float8_e5m2 ->
        let f =
          Contract_cases.cast_to (D.Any D.Float32)
            (Contract_cases.host (A.Any x))
        in
        Array.exists Float.is_nan (A.to_array (A.expect D.Float32 f))
    | _ -> false
  in
  cover "a float8 NaN code, cast"
    ((cast_a && nan_code a) || (cast_b && nan_code b));
  if not (cast_a || cast_b) then
    equal ~msg:"no cast where no operand is cast" (list string) []
      (List.filter_map
         (fun (t : Plan.traced) ->
           let name = Nx_metal.Kernels.name t.kernel in
           if String.starts_with ~prefix:"loop_" name then Some name else None)
         launches);
  equal ~msg:"the vector kernels" bool (ran cast_launches) (ran launches);
  equal (array string) want got

(* Calls that reach the vector kernels at 1 and 16 rows once cast, a weight
   broadcast along a batch of 64, each tile and stage, both layouts of b,
   flipped and stepped operands, and integers past 2^24 and at their
   extremes. *)
let mixed_examples =
  let open D in
  let f32 = Any Float32 in
  let int8_rows = sspec_of ~sexp:false ~sdt:(Any Float16) [ 0; 0; 2 ] in
  [
    scbits_of (cvbits_of (Any Float8_e4m3fn) (Any Float16) 1 1 512 256);
    scbits_of
      (cvbits_of ~vnk:true (Any Float8_e4m3fn) (Any Float16) 1 1 512 256);
    scbits_of (cvbits_of (Any Int32) f32 1 1 300 64);
    scbits_of (cvbits_of ~vnk:true (Any Int8) (Any Bfloat16) 1 2 256 96);
    scbits_of ~ssb:int8_rows
      (cvbits_of ~vnk:true (Any Int8) (Any Int8) 1 16 256 64);
    scbits_of ~ssb:int8_rows
      (cvbits_of ~vnk:true (Any Uint8) (Any Int8) 1 1 256 64);
    scbits_of
      (cvbits_of ~vview:Broadcast_b (Any Float16) (Any Float8_e4m3fn) 64 8 64 40);
    scbits_of
      (cvbits_of ~vview:Broadcast_b ~vnk:true (Any Bfloat16) (Any Int8) 64 8 64
         40);
    scbits_of (cvbits_of (Any Float8_e4m3fn) (Any Float16) 6 256 256 40);
    scbits_of (cvbits_of ~vnk:true (Any Int8) (Any Bfloat16) 1 100 128 40);
    scbits_of (cvbits_of (Any Int64) f32 1 16 64 40);
    scbits_of (cvbits_of (Any Int4) (Any Bfloat16) 1 200 9 100);
    scbits_of (cvbits_of ~vview:Flipped (Any Int64) (Any Uint64) 2 30 40 50);
    scbits_of (cvbits_of ~vview:Stepped_b ~vnk:true (Any Uint32) f32 1 40 70 90);
    scbits_of (cvbits_of ~vat:true (Any Bit) (Any Float4_e2m1fn) 2 40 70 90);
    scbits_of ~ssa:(sspec_of [ 0; 0; 2 ]) (cvbits_of f32 (Any Int16) 1 40 64 64);
  ]

(* Pieces *)

module Pc = Nx_metal.Pieces

type pcase = {
  paxes : Pc.axis array;
  pbatch : int;
  prows : int;
  pa : int;
  pb : int;
  pw : int;
}

let pp_pcase ppf c =
  let axis (x : Pc.axis) =
    Printf.sprintf "%d%s%s%s" x.extent
      (if x.a then "a" else "")
      (if x.b then "b" else "")
      (if x.step > 1 then Printf.sprintf "/%d" x.step else "")
  in
  Format.fprintf ppf "axes [%s], %d batch, %d rows, lines %d and %d, w %d"
    (String.concat "; " (Array.to_list (Array.map axis c.paxes)))
    c.pbatch c.prows c.pa c.pb c.pw

let pcase =
  Gen.(
    with_pp pp_pcase
      (let* pbatch, prows, pcols =
         triple (int_range 0 2) (int_range 0 2) (int_range 0 2)
       in
       let r = pbatch + prows + pcols in
       let axis i (extent, va, vb, step) =
         let batch = i < pbatch in
         {
           Pc.extent;
           a = (if batch then va else i < pbatch + prows);
           b = (if batch then vb else i >= pbatch + prows);
           step = (if batch then step else 1);
         }
       in
       let drawn = quad (int_range 1 6) bool bool (int_range 1 3) in
       let+ paxes =
         map
           (fun l -> Array.mapi axis (Array.of_list l))
           (list ~size:(constant r) drawn)
       and+ pa = int_range 0 40
       and+ pb = int_range 0 40
       and+ pw = int_range 1 400 in
       { paxes; pbatch; prows; pa; pb; pw }))

(* Pieces.mli: the boxes partition the outputs, a range of more than one index
   along an axis of scale blocks keeps to whole blocks, the bytes stated are the
   boxes' most, and every box fits [w] where a line of each operand takes at
   most half of it; else a box past [w] is one batch element, in which an
   operand of several lines takes at most half of [w]. *)
let law_cut_pieces c =
  let axes = c.paxes in
  let r = Array.length axes in
  let p = Pc.cut ~w:c.pw ~batch:c.pbatch ~rows:c.prows axes ~a:c.pa ~b:c.pb in
  let outputs = Array.fold_left (fun n (x : Pc.axis) -> n * x.extent) 1 axes in
  let seen = Array.make outputs 0 in
  let number idx =
    let n = ref 0 in
    Array.iteri (fun i v -> n := (!n * axes.(i).extent) + v) idx;
    !n
  in
  Array.iter
    (fun (x : Pc.box) ->
      let idx = Array.copy x.lo in
      let rec walk i =
        if i = r then seen.(number idx) <- seen.(number idx) + 1
        else
          for v = x.lo.(i) to x.lo.(i) + x.len.(i) - 1 do
            idx.(i) <- v;
            walk (i + 1)
          done
      in
      walk 0;
      Array.iteri
        (fun i (ax : Pc.axis) ->
          let lo = x.lo.(i) and len = x.len.(i) in
          if len > 1 && ax.step > 1 then begin
            equal ~msg:"a range's start on a block" int 0 (lo mod ax.step);
            equal ~msg:"a range's end on a block or the extent" bool true
              ((lo + len) mod ax.step = 0 || lo + len = ax.extent)
          end)
        axes)
    p.boxes;
  equal ~msg:"each output in one box" (array int) (Array.make outputs 1) seen;
  let sizes = Array.map (Pc.bytes axes ~a:c.pa ~b:c.pb) p.boxes in
  let most f = Array.fold_left (fun m s -> Int.max m (f s)) 0 sizes in
  equal ~msg:"a's bytes" int (most fst) p.a_bytes;
  equal ~msg:"b's bytes" int (most snd) p.b_bytes;
  let total (x, y) = x + y in
  let past =
    List.filter
      (fun (_, s) -> total s > c.pw)
      (Array.to_list (Array.combine p.boxes sizes))
  in
  let within_half = 2 * c.pa <= c.pw && 2 * c.pb <= c.pw in
  let lines (x : Pc.box) pick =
    let n = ref 1 in
    for i = c.pbatch to r - 1 do
      if pick axes.(i) then n := !n * x.len.(i)
    done;
    !n
  in
  let several x line pick = line > 0 && lines x pick > 1 in
  let in_a ax = ax.Pc.a and in_b ax = ax.Pc.b in
  if within_half then
    equal ~msg:"the boxes past w" (list int) []
      (List.map (fun (_, s) -> total s) past)
  else
    (* Else a box past w is one batch element, and each cast operand of more
       than one line in it takes at most w / 2: so at least one is one line,
       since two of at most w / 2 would fit w. *)
    List.iter
      (fun ((x : Pc.box), (sa, sb)) ->
        equal ~msg:"a box past w, its batch" (array int) (Array.make c.pbatch 1)
          (Array.sub x.len 0 c.pbatch);
        let at_half what line bytes pick =
          if several x line pick then
            at_most
              ~msg:("a box past w, " ^ what ^ " of several lines")
              int ~than:(c.pw / 2) bytes
        in
        at_half "a" c.pa sa in_a;
        at_half "b" c.pb sb in_b)
      past;
  let n = Array.length p.boxes in
  let whole_free (x : Pc.box) =
    let ok = ref true in
    for i = c.pbatch to r - 1 do
      if x.len.(i) <> axes.(i).extent then ok := false
    done;
    !ok
  in
  let cut_in lo hi (x : Pc.box) =
    let cut = ref false in
    for i = lo to hi - 1 do
      if x.len.(i) <> axes.(i).extent then cut := true
    done;
    !cut
  in
  let rows_end = c.pbatch + c.prows in
  cover "one box" (n = 1);
  cover "boxes of whole batch elements"
    (n > 1 && Array.for_all whole_free p.boxes);
  cover "one operand's axes cut"
    (Array.exists
       (fun x -> cut_in c.pbatch rows_end x <> cut_in rows_end r x)
       p.boxes);
  cover "both operands' axes cut"
    (Array.exists
       (fun x -> cut_in c.pbatch rows_end x && cut_in rows_end r x)
       p.boxes);
  cover "lines within half of w, cut" (within_half && n > 1);
  cover "a box past w" (past <> []);
  cover "a box past w, an operand of several lines"
    (List.exists
       (fun (x, _) -> several x c.pa in_a || several x c.pb in_b)
       past);
  cover "a range of scale blocks"
    (Array.exists
       (fun (x : Pc.box) ->
         let hit = ref false in
         Array.iteri
           (fun i (ax : Pc.axis) ->
             if ax.step > 1 && x.len.(i) > 1 && x.len.(i) < ax.extent then
               hit := true)
           axes;
         !hit)
       p.boxes);
  cover "a last range shorter"
    (n > 1
    && Array.exists (fun (x : Pc.box) -> x.len <> p.boxes.(0).len) p.boxes)

(* A box past w of a's three lines, its whole cast within w / 2, beside one line
   of b past what w leaves. *)
let pcase_examples =
  let axis extent a b = { Pc.extent; a; b; step = 1 } in
  [
    {
      paxes = [| axis 3 true false; axis 2 false true |];
      pbatch = 0;
      prows = 1;
      pa = 5;
      pb = 40;
      pw = 30;
    };
  ]

(* Mixed calls in pieces *)

let cut_parts = [| 0; 2; 3; 5; 9; -64 |]

(* The distinct elements of [x]: an axis of stride 0 holds one. *)
let distinct_axes (A.Any x) pick =
  let l = A.layout x in
  let n = ref 1 in
  for i = 0 to A.Layout.rank l - 1 do
    if pick i && A.Layout.stride l i <> 0 then n := !n * A.Layout.dim l i
  done;
  !n

let varies (A.Any x) i = A.Layout.stride (A.layout x) i <> 0
let stage_bytes (D.Any d) = D.bits d / 8

(* The buffer a test cuts a call's casts within: a [p]th of them, or for [p = 0]
   half the larger line, so that a line passes it, and for [p < 0] a [-p]th of
   it, so that a band of the line is at its floor of a span of terms a range;
   doubled while the pieces number more than [most], so that a run stays
   short. *)
let pieces_w ~p ~most ~total ~lines cut =
  let w0 =
    if p = 0 then Int.max 1 (lines / 2)
    else if p < 0 then Int.max 1 (lines / -p)
    else Int.max 1 (total / p)
  in
  let rec grow w =
    if Array.length (cut w).Pc.boxes <= most || w >= total then w
    else grow (2 * w)
  in
  grow w0

(* [f ()] with a call's casts cut within [w] bytes. *)
let pieced w f =
  Nx_metal.bound_pieces w;
  Fun.protect ~finally:(fun () -> Nx_metal.bound_pieces Plan.kept_workspace) f

(* [f ()] with at most [n] outputs a contraction's piece. *)
let capped n f =
  Nx_metal.bound_outputs n;
  Fun.protect
    ~finally:(fun () ->
      Nx_metal.bound_outputs Nx_metal.Contract.largest_outputs)
    f

(* The contraction plans [launches] ran: each plan's launch of its first tiles
   and first chunk, on the tiles or the vector kernels. *)
let contraction_plans launches =
  let module K = Nx_metal.Kernels in
  let tiles =
    [
      "contract_f32_";
      "contract_f16_";
      "contract_bf16_";
      "contract_i8_";
      "contract_c64_";
      "contract_rows_";
      "contract_ld_";
    ]
  in
  let first (t : Plan.traced) =
    if tile_prefix "contract_mv_" t.kernel then word t K.mv_first = 0
    else if List.exists (fun p -> tile_prefix p t.kernel) tiles then
      word t K.contract_first = 0 && word t K.contract_chunk = 0
    else false
  in
  List.length (List.filter first launches)

(* The bands nx_metal.mli's Contraction gives each box of [sizes] (its casts'
   bytes of a and of b) within [w]: none where the vector kernels take the call
   ([vector]) or the box fits [w]; else each cast line past its share, what the
   other's cast leaves where that takes at most [w / 2], else [w / 2]. Only one
   line of [lines] (a's and b's bytes) of an operand [can] band is banded; the
   other line's share beside one cast whole is what that leaves, a byte at
   least. *)
let band_shares ?(can = (true, true)) ~vector ~w ~lines sizes =
  let la, lb = lines and can_a, can_b = can in
  let one s l = l > 0 && s <= l in
  let share s = Int.max 1 (w - s) in
  Array.map
    (fun (sa, sb) ->
      let band_a = can_a && one sa la and band_b = can_b && one sb lb in
      if vector || sa + sb <= w then (0, 0)
      else if not band_a then if band_b then (0, share sa) else (0, 0)
      else if not band_b then (share sb, 0)
      else if 2 * sa <= w then (0, w - sa)
      else if 2 * sb <= w then (w - sb, 0)
      else (w / 2, w / 2))
    sizes

type bands = {
  band_launches : int;
  most_ranges : int;
  most_chunk : int;
  floor : bool;
}

(* The band launches of [launches], whose pieces' shares are [shares] in order,
   checked: a tile launch reads a band where its piece has one, after the band
   launches of its chunk; a band of more than a span of terms a range takes at
   most its piece's share, at [width] bytes a term. A piece's plan starts at its
   tile launch of first window and chunk 0, which its first band launches
   precede. *)
let band_order ~width shares launches =
  let module K = Nx_metal.Kernels in
  let tiles =
    [
      "contract_f32_";
      "contract_f16_";
      "contract_bf16_";
      "contract_rows_";
      "contract_ld_";
    ]
  in
  let piece = ref (-1) and chunk = ref (-1) and length = ref 0 in
  let r =
    ref { band_launches = 0; most_ranges = 0; most_chunk = 0; floor = false }
  in
  let share i =
    let ba, bb = shares.(i) in
    Int.max ba bb
  in
  List.iter
    (fun (t : Plan.traced) ->
      if Nx_metal.Kernels.name t.kernel = "contract_band" then begin
        let c = word t K.band_chunk and l = word t K.band_length in
        let i = if c = 0 then !piece + 1 else !piece in
        let _, ranges, g = t.groups in
        let bytes = ranges * g * l * width in
        at_most ~msg:"a band launch in a piece of bands" int
          ~than:(Array.length shares - 1)
          i;
        less ~msg:"a band launch in a piece of bands" int ~than:(share i) 0;
        if l > 32 then
          at_most ~msg:"a band within its share" int ~than:(share i) bytes;
        chunk := c;
        length := l;
        r :=
          {
            band_launches = !r.band_launches + 1;
            most_ranges = Int.max !r.most_ranges ranges;
            most_chunk = Int.max !r.most_chunk c;
            floor = !r.floor || bytes > share i;
          }
      end
      else if List.exists (fun p -> tile_prefix p t.kernel) tiles then begin
        if word t K.contract_first = 0 && word t K.contract_chunk = 0 then
          incr piece;
        let codes = word t K.contract_codes in
        let bit i = (codes lsr i) land 1 = 1 in
        let banded = bit K.codes_band_a || bit K.codes_band_b in
        equal ~msg:"a tile reads a band where its piece has one" bool
          (share !piece > 0)
          banded;
        equal ~msg:"a tile reads a band only as a banded tile" bool banded
          (banded_twin t.kernel);
        if banded then begin
          equal ~msg:"a tile's chunk is its band's" int !chunk
            (word t K.contract_chunk);
          equal ~msg:"a tile's terms are its band's" int !length
            (word t K.contract_length)
        end
      end)
    launches;
  !r

let scale_block shape s i =
  match s with
  | None -> 1
  | Some (A.Any x) ->
      let es = A.Layout.dim (A.layout x) i in
      if es = 1 || es = List.nth shape i then 1 else List.nth shape i / es

let scale_any = function
  | None -> None
  | Some (Nx_kernel.Exponent e) -> Some (A.Any e)
  | Some (Nx_kernel.Value v) -> Some (A.Any v)

(* nx_metal.mli: a call whose casts pass the buffer is cut into pieces that keep
   the call's exchange, tiles, ranges and vector kernels: every output has the
   bits of the call uncut. *)
let law_mixed_pieces (w, p) =
  let c = w.sc in
  let bk = backend () in
  let st = Random.State.make [| c.vseed |] in
  let bt = c.vbatch and m = c.vm and n = c.vn and k = c.vk in
  let a, _, b, _ = cv_operands st bk c in
  let a_shape = [ bt; m; k ] in
  let b_shape = if c.vnk then [ bt; n; k ] else [ bt; k; n ] in
  let scale sp shape = fst (drawn_scale st bk sp shape) in
  let sa = Option.map (fun sp -> scale sp a_shape) w.ssa in
  let sb = Option.map (fun sp -> scale sp b_shape) w.ssb in
  let scaled = Option.is_some sa || Option.is_some sb in
  let stage = stage_dtype ~scaled c.va c.vb in
  let shape = [ bt; m; n ] in
  let (D.Any ddt) = c.vdst in
  let init =
    if not c.vinit then None
    else
      let name = D.name ddt and outputs = bt * m * n in
      let vs = drawn_values st ~name ~low:c.vlow ~special:16 outputs in
      let made = A.of_array D.Float64 (Iarray.of_list shape) vs in
      let x = Contract_cases.cast_to c.vdst (A.Any made) in
      Some (Support.on bk (A.expect ddt x))
  in
  let run () =
    let dst = A.create (S.device ()) ddt (Iarray.of_list shape) in
    let got, launches =
      Plan.traced @@ fun () ->
      Nx_metal.contract
        ~batch:(Iarray.of_list [ (0, 0) ])
        ~contracting:(Iarray.of_list [ (2, if c.vnk then 2 else 1) ])
        ~segments:None ~acc:(D.Any D.Float32) ~dst ~init
        { Nx_kernel.array = a; scale = sa }
        { Nx_kernel.array = b; scale = sb }
    in
    equal answer A.Done got;
    (Support.elements (A.Any dst), launches)
  in
  (* The vector kernels and the loaders read their operands at their own dtypes,
     so neither is cast. *)
  let own =
    vector_own ~m ~n ~k c.va sa c.vb sb
    || loader_own ~bt ~m ~n ~k ~nk:c.vnk (a, c.va, w.ssa) (b, c.vb, w.ssb)
  in
  let cast d s = (not own) && Option.is_none s && d <> stage in
  let cast_a = cast c.va sa and cast_b = cast c.vb sb in
  let sbytes = stage_bytes stage in
  let b_cols = if c.vnk then 1 else 2 and b_terms = if c.vnk then 2 else 1 in
  let line cast x terms =
    if cast then sbytes * distinct_axes x (fun i -> i = terms) else 0
  in
  let line_a = line cast_a a 2 and line_b = line cast_b b b_terms in
  let step =
    let blk s sh = scale_block sh (scale_any s) 0 in
    let p = blk sa a_shape and q = blk sb b_shape in
    p * q
  in
  let axes =
    [|
      {
        Pc.extent = bt;
        a = cast_a && varies a 0;
        b = cast_b && varies b 0;
        step;
      };
      { Pc.extent = m; a = cast_a && varies a 1; b = false; step = 1 };
      { Pc.extent = n; a = false; b = cast_b && varies b b_cols; step = 1 };
    |]
  in
  let cut w = Pc.cut ~w ~batch:1 ~rows:1 axes ~a:line_a ~b:line_b in
  let total =
    (if cast_a then sbytes * distinct_axes a (fun _ -> true) else 0)
    + if cast_b then sbytes * distinct_axes b (fun _ -> true) else 0
  in
  let w_bytes =
    pieces_w ~p:cut_parts.(p) ~most:400 ~total ~lines:(Int.max line_a line_b)
      cut
  in
  let want, _ = run () in
  let got, launches = pieced w_bytes run in
  let boxes = (cut w_bytes).boxes in
  let plans = contraction_plans launches in
  equal ~msg:"a contraction plan a piece" int
    (if k = 0 then 0 else Array.length boxes)
    plans;
  let cut_into = plans > 1 in
  let ran =
    List.exists
      (fun (t : Plan.traced) -> tile_prefix "contract_mv_" t.kernel)
      launches
  in
  let e =
    {
      C.batch = bt;
      m;
      n;
      k;
      stage = C.F32;
      nk = false;
      complex = false;
      source = 0;
    }
  in
  let ranges = match C.plan e with Some p -> p.ranges | None -> 1 in
  let sizes = Array.map (Pc.bytes axes ~a:line_a ~b:line_b) boxes in
  let shares =
    band_shares ~vector:ran ~w:w_bytes ~lines:(line_a, line_b) sizes
  in
  let bands = band_order ~width:sbytes shares launches in
  let banded f = Array.exists f shares in
  cover "a banded" (banded (fun (x, y) -> x > 0 && y = 0));
  cover "b banded" (banded (fun (x, y) -> x = 0 && y > 0));
  cover "both banded" (banded (fun (x, y) -> x > 0 && y > 0));
  cover "a band of several ranges" (bands.most_ranges > 1);
  cover "a band of several chunks" (bands.most_chunk > 0);
  cover "a band at the span floor" bands.floor;
  (* band_order checks that a tile reads a band exactly where it is a banded
     twin. *)
  let twin (t : Plan.traced) = banded_twin t.kernel in
  let twins f =
    List.exists
      (fun (t : Plan.traced) ->
        twin t && f (is_loader t.kernel) (Nx_metal.Kernels.name t.kernel))
      launches
  in
  let prefix p ld n = (not ld) && String.starts_with ~prefix:p n in
  let suffix x ld n = (not ld) && String.ends_with ~suffix:x n in
  cover "a float32 tile's twin"
    (bands.band_launches > 0 && twins (prefix "contract_f32_"));
  cover "a float16 tile's twin" (twins (prefix "contract_f16_"));
  cover "a bfloat16 tile's twin" (twins (prefix "contract_bf16_"));
  cover "an n x k tile's twin" (twins (suffix "_nk_band"));
  cover "a narrow tile's twin" (twins (suffix "_kn_n_band"));
  cover "a loader's twin" (twins (fun ld _ -> ld));
  cover "a cut call without a band, of no twin"
    (cut_into && bands.band_launches = 0 && not (List.exists twin launches));
  cover "a vector call past the buffer, its lines cast whole"
    (ran && Array.exists (fun (x, y) -> x + y > w_bytes) sizes);
  cover "a cast, cut" (cut_into && cast_a && not cast_b);
  cover "b cast, cut" (cut_into && cast_b && not cast_a);
  cover "both cast, cut" (cut_into && cast_a && cast_b);
  cover "both cast past half the buffer"
    (cut_into && cast_a && cast_b
    && 2 * line_a * m > w_bytes
    && 2 * line_b * n > w_bytes);
  cover "cut along the batch"
    (cut_into && Array.exists (fun (x : Pc.box) -> x.len.(0) < bt) boxes);
  cover "exchanged, cut" (cut_into && n <= 16 && m > 16);
  cover "the vector kernels, cut" (cut_into && ran);
  cover "a vector piece of fewer than 256 columns"
    (cut_into && ran && Array.exists (fun (x : Pc.box) -> x.len.(2) < 256) boxes);
  cover "a piece of 16 columns or fewer"
    (cut_into
    && Array.exists (fun (x : Pc.box) -> x.len.(2) <= 16 && x.len.(2) < n) boxes
    );
  cover "a broadcast weight, cut" (cut_into && c.vview = Broadcast_b && cast_b);
  cover "a scaled operand beside a cast, cut along the batch"
    (cut_into && scaled
    && Array.exists (fun (x : Pc.box) -> x.len.(0) < bt) boxes);
  cover "several ranges, cut" (cut_into && ranges > 1);
  cover "a line past the buffer"
    (cut_into && Array.exists (fun (x, y) -> x + y > w_bytes) sizes);
  equal (array string) want got

(* nx_metal.mli: a cast line past the buffer whose terms make several runs of
   strides is cast whole, and the call keeps its bits. a is [1; k1; k2] of
   strides [k1 k2; 1; k1]: its terms do not merge. Beside a b not cast, no band
   runs; beside a cast b ([cast_b]) of one run past what a's line leaves, b
   alone is banded within it, so a's line and b's band take at most w
   together. *)
let band_runs ~cast_b () =
  let bk = backend () in
  let st = Random.State.make [| 7 |] in
  let m = 1 and k1 = 40 and k2 = 30 and n = 3 in
  let k = k1 * k2 in
  let host dt shape name =
    let count = List.fold_left ( * ) 1 shape in
    let vs = drawn_values st ~name ~low:0 ~special:16 count in
    let made = A.of_array D.Float64 (Iarray.of_list shape) vs in
    Contract_cases.cast_to dt (A.Any made)
  in
  let a0 =
    Support.on bk
      (A.expect D.Int32 (host (D.Any D.Int32) [ m; k2; k1 ] "int32"))
  in
  let a =
    A.v D.Int32
      (A.Layout.v
         ~offset:(A.Layout.offset (A.layout a0))
         ~strides:(Iarray.of_list [ k; 1; k1 ])
         (Iarray.of_list [ m; k1; k2 ]))
      (A.buffer a0)
  in
  let b =
    if cast_b then
      let x =
        Support.on bk
          (A.expect D.Int32 (host (D.Any D.Int32) [ k1; k2; n ] "int32"))
      in
      A.Any x
    else
      let x =
        Support.on bk
          (A.expect D.Float32 (host (D.Any D.Float32) [ k1; k2; n ] "float32"))
      in
      A.Any x
  in
  let run () =
    let dst = A.create (S.device ()) D.Float32 (Iarray.of_list [ m; n ]) in
    let got, launches =
      Plan.traced @@ fun () ->
      Nx_metal.contract ~batch:(Iarray.of_list [])
        ~contracting:(Iarray.of_list [ (1, 0); (2, 1) ])
        ~segments:None ~acc:(D.Any D.Float32) ~dst ~init:None
        { Nx_kernel.array = A.Any a; scale = None }
        { Nx_kernel.array = b; scale = None }
    in
    equal answer A.Done got;
    (Support.elements (A.Any dst), launches)
  in
  let line = 4 * k in
  let w = if cast_b then 3 * line / 2 else line / 2 in
  let want, _ = run () in
  let got, launches = pieced w run in
  let lb = if cast_b then line else 0 in
  (* Beside b uncast, a's one line is one box; else each box is a's line and one
     column of b, both past w / 2. *)
  let sizes = if cast_b then Array.make n (line, lb) else [| (line, 0) |] in
  let shares =
    band_shares ~can:(false, true) ~vector:false ~w ~lines:(line, lb) sizes
  in
  let bands = band_order ~width:4 shares launches in
  let module K = Nx_metal.Kernels in
  let a_banded =
    List.exists
      (fun (t : Plan.traced) ->
        tile_prefix "contract_f32_" t.kernel
        && (word t K.contract_codes lsr K.codes_band_a) land 1 = 1)
      launches
  in
  equal ~msg:"a's line banded" bool false a_banded;
  equal ~msg:"b banded" bool cast_b (bands.band_launches > 0);
  equal ~msg:"a band at its floor" bool false bands.floor;
  equal ~msg:"a contraction plan a box" int (Array.length sizes)
    (contraction_plans launches);
  equal (array string) want got

let pp_pieced pp ppf (x, p) =
  Format.fprintf ppf "%a, cut in %d" pp x cut_parts.(p)

let mixed_pieces_case =
  Gen.(with_pp (pp_pieced pp_scbits) (pair scbits_case (int_range 0 4)))

(* Calls that cut a, b and both, along the batch, exchanged, on the vector
   kernels below 256 columns, beside a scaled weight, of several ranges, and
   past one line; and banded calls on the float16, bfloat16, n x k and narrow
   tiles' twins. *)
let mixed_pieces_examples =
  let open D in
  let f32 = Any Float32 in
  let int8_rows = sspec_of ~sexp:false ~sdt:(Any Float16) [ 0; 0; 2 ] in
  [
    (scbits_of (cvbits_of (Any Float16) (Any Float8_e4m3fn) 1 64 300 96), 2);
    ( scbits_of
        (cvbits_of ~vnk:true (Any Float8_e4m3fn) (Any Bfloat16) 1 200 64 96),
      3 );
    (scbits_of (cvbits_of (Any Int8) (Any Float8_e5m2) 1 100 200 64), 4);
    (scbits_of (cvbits_of (Any Int8) (Any Bfloat16) 6 64 64 40), 2);
    (scbits_of (cvbits_of (Any Float8_e4m3fn) (Any Float16) 1 300 8 64), 3);
    (scbits_of (cvbits_of (Any Float16) (Any Float8_e4m3fn) 1 1 600 512), 3);
    (scbits_of (cvbits_of (Any Float16) (Any Float8_e4m3fn) 1 1 300 64), 4);
    ( scbits_of
        (cvbits_of ~vview:Broadcast_b (Any Float16) (Any Float8_e4m3fn) 64 8 64
           40),
      3 );
    ( scbits_of ~ssb:int8_rows
        (cvbits_of ~vnk:true (Any Float8_e4m3fn) (Any Int8) 4 64 64 64),
      2 );
    (scbits_of (cvbits_of f32 (Any Int16) 1 16 64 2048), 3);
    (scbits_of (cvbits_of (Any Int64) f32 1 40 100 64), 0);
    (scbits_of (cvbits_of ~vview:Flipped (Any Int64) (Any Uint64) 2 30 40 50), 1);
    (scbits_of (cvbits_of (Any Int64) f32 1 2 3 9000), 0);
    (scbits_of (cvbits_of (Any Int64) f32 1 2 3 9000), 5);
    (scbits_of (cvbits_of f32 (Any Int32) 1 20 3 9000), 0);
    (scbits_of (cvbits_of (Any Int64) (Any Uint32) 1 1 1 9000), 0);
    (scbits_of (cvbits_of (Any Int4) (Any Bfloat16) 1 3 4 9001), 0);
    (scbits_of (cvbits_of (Any Float16) (Any Float8_e4m3fn) 1 1 300 64), 0);
    ( scbits_of ~ssb:int8_rows
        (cvbits_of ~vnk:true (Any Float8_e4m3fn) (Any Int8) 1 64 64 2048),
      0 );
    (scbits_of (cvbits_of (Any Float16) (Any Float8_e4m3fn) 1 64 64 512), 0);
    (scbits_of (cvbits_of (Any Bfloat16) (Any Float8_e4m3fn) 1 64 64 512), 0);
    (scbits_of (cvbits_of ~vnk:true (Any Float8_e4m3fn) f32 1 64 64 512), 0);
    (scbits_of (cvbits_of (Any Float8_e4m3fn) f32 1 64 40 512), 0);
  ]

(* A call in pieces *)

type dkind =
  | Dplain of D.any
  | Dcomplex
  | Dbytes of D.any
  | Dwide
  | Dmixed
  | Dvector
  | Drows

type dview = Cols_apart | Cols_swapped | Merged

type dcase = {
  dkind : dkind;
  dbt : int;
  dm : int;
  dn1 : int;
  dn2 : int;
  dk : int;
  dg : int;  (** A Rows call's groups. *)
  dview : dview;
  dinit : bool;
  dparts : int;
  dcap : int;  (** The outputs' share a piece takes at most, [0] for none. *)
}

let pp_dkind ppf = function
  | Dplain (D.Any d) -> Format.fprintf ppf "%s" (D.name d)
  | Dcomplex -> Format.fprintf ppf "complex64"
  | Dbytes (D.Any d) -> Format.fprintf ppf "int8 into %s" (D.name d)
  | Dwide -> Format.fprintf ppf "int32"
  | Dmixed -> Format.fprintf ppf "float16 x int16"
  | Dvector -> Format.fprintf ppf "a float16 weight"
  | Drows -> Format.fprintf ppf "Rows"

let pp_dcase ppf c =
  Format.fprintf ppf "%a, %d x %d x (%d x %d), k %d%s, %s%s, a %dth%s" pp_dkind
    c.dkind c.dbt c.dm c.dn1 c.dn2 c.dk
    (if c.dkind = Drows then Printf.sprintf ", %d groups" c.dg else "")
    (match c.dview with
    | Cols_apart -> "columns apart"
    | Cols_swapped -> "columns swapped"
    | Merged -> "C-contiguous")
    (if c.dinit then ", init" else "")
    c.dparts
    (if c.dcap = 0 then "" else Printf.sprintf ", outputs a %dth" c.dcap)

(* The dtypes of [c]'s a, b and dst, and its accumulator. *)
let dtypes c =
  let open D in
  let f32 = Any Float32 and i32 = Any Int32 in
  match c.dkind with
  | Dplain d -> (d, d, d, f32)
  | Dcomplex -> (Any Complex64, Any Complex64, Any Complex64, Any Complex64)
  | Dbytes d -> (Any Int8, Any Int8, d, i32)
  | Dwide -> (i32, i32, i32, i32)
  | Dmixed -> (Any Float16, Any Int16, f32, f32)
  | Dvector -> (f32, Any Float16, f32, f32)
  | Drows -> (f32, f32, f32, f32)

let dcase =
  let open Gen in
  with_pp pp_dcase
    (let* dkind =
       frequency
         [
           ( 2,
             map
               (fun d -> Dplain d)
               (of_list ~pp:Contract_cases.pp_dtype
                  D.[ Any Float32; Any Float16; Any Bfloat16 ]) );
           (1, constant Dcomplex);
           ( 1,
             map
               (fun d -> Dbytes d)
               (of_list ~pp:Contract_cases.pp_dtype
                  D.[ Any Int32; Any Int8; Any Int4 ]) );
           (1, constant Dwide);
           (1, constant Dmixed);
           (1, constant Dvector);
           (1, constant Drows);
         ]
     in
     let m =
       match dkind with
       | Dvector -> int_range 1 2
       | Dmixed -> int_range 17 40
       | _ -> int_range 1 40
     in
     let k =
       match dkind with
       | Dvector -> map (fun q -> 64 * q) (int_range 1 4)
       | _ -> int_range 0 70
     in
     (* The vector kernels take 256 columns or more. *)
     let n1, n2 =
       match dkind with
       | Dvector -> (int_range 4 8, constant 64)
       | _ -> (int_range 2 9, int_range 2 9)
     in
     let+ dbt = int_range 1 3
     and+ dm = m
     and+ dn1 = n1
     and+ dn2 = n2
     and+ dk = k
     and+ dg = int_range 1 4
     and+ dview = of_list [ Cols_apart; Cols_swapped; Merged ]
     and+ dinit = bool
     and+ dparts = of_list [ 2; 3; 7 ]
     and+ dcap = frequency [ (1, constant 0); (2, of_list [ 2; 5 ]) ] in
     {
       dkind;
       dbt = (if dkind = Drows then 1 else dbt);
       dm;
       dn1;
       dn2;
       dk;
       dg;
       dview;
       dinit;
       dparts;
       dcap;
     })

(* An array of [dt] and [shape] on the GPU of values [f i] at C index [i]. *)
let valued dt shape f =
  let n = List.fold_left ( * ) 1 shape in
  let made =
    A.of_array D.Float64 (Iarray.of_list shape)
      (Array.init n (fun i -> Float.of_int (f i)))
  in
  Contract_cases.on (backend ()) (Contract_cases.cast_to dt (A.Any made))

(* [c]'s Rows ids: runs of each group in increasing order, a run of length 1 to
   3. *)
let row_ids c =
  let ids = Array.make c.dm 0 in
  let r = ref 0 and g = ref 0 in
  while !r < c.dm do
    let len = 1 + ((!r + !g) mod 3) in
    for i = !r to Int.min c.dm (!r + len) - 1 do
      ids.(i) <- !g mod c.dg
    done;
    r := !r + len;
    incr g
  done;
  ids

(* [c]'s dst shape, the batch first but in a Rows call. *)
let dshape c =
  (if c.dkind = Drows then [] else [ c.dbt ]) @ [ c.dm; c.dn1; c.dn2 ]

(* [c]'s dst of [ddt] through its view over an array of 7s, and that array. *)
let dst_apart c ddt =
  let dd = D.Any ddt in
  let shape = dshape c in
  let lead = List.filteri (fun i _ -> i < List.length shape - 2) shape in
  let made =
    match c.dview with
    | Cols_apart -> lead @ [ c.dn1; c.dn2 + 1 ]
    | Cols_swapped -> lead @ [ c.dn2; c.dn1 ]
    | Merged -> shape
  in
  let (A.Any whole) = valued dd made (fun _ -> 7) in
  let whole = A.expect ddt (A.Any whole) in
  let c_strides s =
    List.fold_right (fun e (acc, st) -> (st :: acc, st * e)) s ([], 1) |> fst
  in
  let st = c_strides made in
  let strides =
    match c.dview with
    | Cols_apart | Merged -> st
    | Cols_swapped ->
        let outer = List.filteri (fun i _ -> i < List.length st - 2) st in
        outer @ [ 1; c.dn1 ]
  in
  let l = A.Layout.v ~strides:(Iarray.of_list strides) (Iarray.of_list shape) in
  (A.v ddt l (A.buffer whole), whole)

(* nx_metal.mli: a call whose dst does not merge its column axes computes in
   pieces, each into a temporary in C order copied into dst, and a call of more
   outputs than {!Nx_metal.bound_outputs}'s in pieces of at most so many: its
   outputs have the bits of the call uncut into a C-contiguous dst, which a copy
   out of order or a piece's own choices would break, and no element outside the
   view changes. Pieces are {!Pc.refine}'s of the call's box, or of {!Pc.cut}'s
   boxes beside casts, within half the buffer each beside a temporary. *)
let law_dst_pieces c =
  let bk = backend () in
  let da, db, (D.Any ddt as dd), acc = dtypes c in
  let rows = c.dkind = Drows in
  let bt = c.dbt and m = c.dm and n1 = c.dn1 and n2 = c.dn2 and k = c.dk in
  let shape = dshape c in
  let a_shape = (if rows then [] else [ bt ]) @ [ m; k ] in
  let b_shape = (if rows then [ c.dg ] else [ bt ]) @ [ k; n1; n2 ] in
  (* float32 values whose products round, so that a sum's association shows in
     its bits. *)
  let rounding =
    match c.dkind with
    | Dcomplex | Dplain (D.Any D.Float32) -> true
    | _ -> false
  in
  let small i =
    let v = (((i * 7) + 3) mod 13) - 6 in
    if rounding then (v * 4099) + (i mod 5) else v
  in
  let a = valued da a_shape small
  and b = valued db b_shape (fun i -> small (i + 5)) in
  let init =
    if not c.dinit then None
    else Some (A.expect ddt (valued dd shape (fun i -> small (i + 2))))
  in
  let ids = row_ids c in
  let segments =
    if not rows then None
    else
      let x =
        A.of_array D.Int64 (Iarray.of_list [ m ]) (Array.map Int64.of_int ids)
      in
      Some (Nx_kernel.Rows { ids = Support.on bk x; a_axis = 0; b_axis = 0 })
  in
  let run dst =
    let got, launches =
      Plan.traced @@ fun () ->
      Nx_metal.contract
        ~batch:(Iarray.of_list (if rows then [] else [ (0, 0) ]))
        ~contracting:(Iarray.of_list (if rows then [ (1, 1) ] else [ (2, 1) ]))
        ~segments ~acc ~dst ~init
        { Nx_kernel.array = a; scale = None }
        { Nx_kernel.array = b; scale = None }
    in
    equal answer A.Done got;
    launches
  in
  let fresh = A.create (S.device ()) ddt (Iarray.of_list shape) in
  ignore (run fresh);
  let want = elements fresh in
  let dst, whole = dst_apart c ddt in
  let before = elements whole in
  let bits = D.bits ddt in
  let outputs = List.fold_left ( * ) 1 shape in
  let w = Int.max 1 (((outputs * bits) + 7) / 8 / c.dparts) in
  let cap =
    if c.dcap = 0 then Nx_metal.Contract.largest_outputs
    else Int.max 1 (outputs / c.dcap)
  in
  let launches = pieced w (fun () -> capped cap (fun () -> run dst)) in
  let got = elements dst in
  (* The positions of [whole] the view covers. *)
  let inside = Array.make (Array.length before) false in
  let l = A.layout dst in
  let r = A.Layout.rank l in
  let idx = Array.make r 0 in
  let rec walk i =
    if i = r then begin
      let p = ref (A.Layout.offset l) in
      Array.iteri (fun j v -> p := !p + (v * A.Layout.stride l j)) idx;
      inside.(!p) <- true
    end
    else
      for v = 0 to A.Layout.dim l i - 1 do
        idx.(i) <- v;
        walk (i + 1)
      done
  in
  walk 0;
  let after = elements whole in
  let changed = ref [] in
  Array.iteri
    (fun p x ->
      if (not inside.(p)) && x <> after.(p) then changed := p :: !changed)
    before;
  equal ~msg:"elements outside the view changed" (list int) [] !changed;
  let axes =
    Array.of_list
      (List.map
         (fun extent -> { Pc.extent; a = false; b = false; step = 1 })
         shape)
  in
  let whole_box =
    { Pc.lo = Array.make (List.length shape) 0; len = Array.of_list shape }
  in
  let temp = c.dview <> Merged in
  let boxes =
    match c.dkind with
    | Dmixed when k > 0 ->
        (* Both cast into float32, a line of k terms each. *)
        let axes =
          [|
            { Pc.extent = bt; a = true; b = true; step = 1 };
            { Pc.extent = m; a = true; b = false; step = 1 };
            { Pc.extent = n1; a = false; b = true; step = 1 };
            { Pc.extent = n2; a = false; b = true; step = 1 };
          |]
        in
        let p =
          Pc.cut
            ~w:(if temp then w / 2 else w)
            ~batch:1 ~rows:1 axes ~a:(4 * k) ~b:(4 * k)
        in
        let most =
          if temp then Int.min cap (8 * (w - (w / 2)) / bits) else cap
        in
        List.concat_map (Pc.refine ~most axes) (Array.to_list p.boxes)
    | _ ->
        Pc.refine
          ~most:(if temp then Int.min cap (8 * w / bits) else cap)
          axes whole_box
  in
  let n = List.length boxes in
  (* A piece's plan starts at its first launch of the tiles, the vector kernels
     or the SIMT kernel. *)
  let simt_firsts =
    List.length
      (List.filter
         (fun (t : Plan.traced) ->
           Nx_metal.Kernels.name t.kernel = "contract_simt"
           && word t Nx_metal.Kernels.simt_first = 0
           && word t Nx_metal.Kernels.simt_chunk = 0)
         launches)
  in
  equal ~msg:"a plan a piece" int
    (if k = 0 then 0 else n)
    (contraction_plans launches + simt_firsts);
  let ran =
    List.exists
      (fun (t : Plan.traced) -> tile_prefix "contract_mv_" t.kernel)
      launches
  in
  let row_cuts =
    List.filter_map
      (fun (x : Pc.box) ->
        let mid = ref false in
        Array.iteri (fun i lo -> if i > 0 && lo > 0 then mid := true) x.lo;
        if x.lo.(0) > 0 && not !mid then Some x.lo.(0) else None)
      boxes
  in
  cover "several pieces" (n > 1);
  cover "a C-contiguous dst in pieces" ((not temp) && n > 1);
  cover "a dst apart in pieces of fewer outputs than its temporary"
    (temp && cap < 8 * w / bits && n > 1);
  cover "a cut mid-row"
    (List.exists (fun (x : Pc.box) -> x.lo.(Array.length x.lo - 1) > 0) boxes);
  cover "float16" (dd = D.Any D.Float16);
  cover "bfloat16" (dd = D.Any D.Bfloat16);
  cover "complex64" (c.dkind = Dcomplex);
  cover "a dst narrower than a byte" (bits < 8);
  cover "the SIMT kernel" (c.dkind = Dwide);
  cover "casts beside the temporary" (c.dkind = Dmixed);
  cover "the vector kernels" ran;
  cover "init" c.dinit;
  cover "no terms" (k = 0);
  cover "a Rows cut inside a run of equal ids"
    (rows && List.exists (fun r -> ids.(r - 1) = ids.(r)) row_cuts);
  cover "a Rows cut at a group boundary"
    (rows && List.exists (fun r -> ids.(r - 1) <> ids.(r)) row_cuts);
  equal (array string) want got

(* Calls of each kind cut into pieces: the vector kernels, Rows cuts in runs and
   at their bounds, a dst narrower than a byte, float16 and bfloat16 dsts, a cut
   mid-row, and complex64 calls whose whole call sums in one range where a piece
   alone would take two. *)
let dst_pieces_examples =
  let c =
    {
      dkind = Dplain (D.Any D.Float32);
      dbt = 1;
      dm = 1;
      dn1 = 8;
      dn2 = 64;
      dk = 256;
      dg = 1;
      dview = Cols_apart;
      dinit = false;
      dparts = 3;
      dcap = 0;
    }
  in
  let ranged =
    {
      c with
      dkind = Dcomplex;
      dm = 640;
      dn1 = 20;
      dn2 = 32;
      dk = 512;
      dcap = 100;
    }
  in
  [
    { c with dkind = Dvector };
    { c with dkind = Dvector; dm = 2; dview = Cols_swapped; dinit = true };
    {
      c with
      dkind = Drows;
      dm = 12;
      dn1 = 2;
      dn2 = 2;
      dk = 9;
      dg = 3;
      dparts = 7;
    };
    {
      c with
      dkind = Dbytes (D.Any D.Int4);
      dbt = 2;
      dm = 5;
      dn1 = 3;
      dn2 = 5;
      dk = 9;
      dparts = 2;
    };
    {
      c with
      dkind = Dmixed;
      dbt = 2;
      dm = 20;
      dn1 = 3;
      dn2 = 4;
      dk = 40;
      dparts = 2;
    };
    {
      c with
      dkind = Dcomplex;
      dbt = 2;
      dm = 9;
      dn1 = 3;
      dn2 = 3;
      dk = 0;
      dinit = true;
    };
    ranged;
    { ranged with dview = Merged };
    {
      c with
      dkind = Dplain (D.Any D.Float16);
      dbt = 2;
      dm = 2;
      dn1 = 3;
      dn2 = 9;
      dk = 9;
      dview = Merged;
      dcap = 5;
    };
    {
      c with
      dkind = Dplain (D.Any D.Bfloat16);
      dm = 7;
      dn1 = 3;
      dn2 = 4;
      dk = 20;
      dview = Cols_swapped;
      dcap = 2;
    };
  ]

(* Pieces.mli: refine partitions its box, each box of at most [most] outputs or
   of one, and a range of more than one index along an axis of a step starts on
   its multiple. *)
let law_refine_pieces (c, most) =
  let axes = c.paxes in
  let r = Array.length axes in
  let p = Pc.cut ~w:c.pw ~batch:c.pbatch ~rows:c.prows axes ~a:c.pa ~b:c.pb in
  List.iter
    (fun (x : Pc.box) ->
      let boxes = Pc.refine ~most axes x in
      let count (y : Pc.box) = Array.fold_left ( * ) 1 y.len in
      equal ~msg:"the boxes' outputs" int (count x)
        (List.fold_left (fun n y -> n + count y) 0 boxes);
      List.iter
        (fun (y : Pc.box) ->
          for i = 0 to r - 1 do
            if
              y.lo.(i) < x.lo.(i) || y.lo.(i) + y.len.(i) > x.lo.(i) + x.len.(i)
            then failf "a box outside its box along %d" i;
            let ax = axes.(i) in
            if y.len.(i) > 1 && ax.step > 1 && y.len.(i) < x.len.(i) then
              equal ~msg:"a range's start on a block" int 0
                (y.lo.(i) mod ax.step)
          done;
          if count y > 1 then
            at_most ~msg:"a box's outputs" int ~than:most (count y))
        boxes;
      (* In order and disjoint: each box starts after the last one's last output
         in C order of x. *)
      let number idx =
        let n = ref 0 in
        Array.iteri (fun i v -> n := (!n * x.len.(i)) + (v - x.lo.(i))) idx;
        !n
      in
      let last (y : Pc.box) =
        number (Array.mapi (fun i lo -> lo + y.len.(i) - 1) y.lo)
      in
      ignore
        (List.fold_left
           (fun prev (y : Pc.box) ->
             less ~msg:"a box after the last" int ~than:(number y.lo) prev;
             equal ~msg:"a box's outputs in one run" int (count y)
               (last y - number y.lo + 1);
             last y)
           (-1) boxes);
      cover "a box cut" (List.length boxes > 1))
    (Array.to_list p.boxes);
  cover "a box of one output" (most < 1)

(* nx_metal.mli: an Exponent code's power of two applied to an element by the
   arithmetic of its bits, each row an element, its code and the product, as a
   call of one term with b 1 into +0 gives it; and a Value's product, rounded
   once under the flush. *)
let scale_rows =
  let max32 = 0x1.fffffep127 in
  [
    (1., `Code 127, 1.);
    (1.5, `Code 126, 0.75);
    (1., `Code 0, 0.);
    (1., `Code 1, 0x1p-126);
    (1., `Code 254, 0x1p127);
    (2., `Code 254, infinity);
    (max32, `Code 127, max32);
    (max32, `Code 128, infinity);
    (-.max32, `Code 128, neg_infinity);
    (0x1p-100, `Code 101, 0x1p-126);
    (0x1.fffffep-101, `Code 101, 0.);
    (0x1p-130, `Code 200, 0.);
    (infinity, `Code 0, infinity);
    (neg_infinity, `Code 254, neg_infinity);
    (infinity, `Code 255, nan);
    (0., `Code 255, nan);
    (nan, `Code 127, nan);
    (0., `Code 254, 0.);
    (3., `Value 1.5, 4.5);
    (0x1.000002p0, `Value 0x1.000002p0, 0x1.000004p0);
    (0x1p-130, `Value infinity, nan);
    (2., `Value 0x1p-127, 0.);
    (0x1p-100, `Value 0x1p-26, 0x1p-126);
    (0x1p-100, `Value 0x1.fffffep-27, 0.);
    (max32, `Value 2., infinity);
    (infinity, `Value 0x1p-127, nan);
    (-0., `Value 0x1p-127, 0.);
  ]

let scale_row_id (x, s, _) =
  match s with
  | `Code e -> Printf.sprintf "%h by code %d" x e
  | `Value v -> Printf.sprintf "%h by value %h" x v

let test_scale_row (x, s, want) =
  let bk = backend () in
  let one v =
    Support.on bk (A.of_array D.Float32 (Iarray.of_list [ 1; 1; 1 ]) [| v |])
  in
  let scale =
    match s with
    | `Code e ->
        Nx_kernel.Exponent
          (Support.on bk
             (A.of_array D.Uint8 (Iarray.of_list [ 1; 1; 1 ]) [| e |]))
    | `Value v -> Nx_kernel.Value (one v)
  in
  let dst = A.create (S.device ()) D.Float32 (Iarray.of_list [ 1; 1; 1 ]) in
  let got =
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (2, 1) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst ~init:None
      { Nx_kernel.array = A.Any (one x); scale = Some scale }
      { Nx_kernel.array = A.Any (one 1.); scale = None }
  in
  equal answer A.Done got;
  equal (array float_exact) [| want |] (A.to_array (Support.host dst))

(* The models' weights *)

(* float4_e2m1fn's value of code [c]: its sign bit, then 0, 0.5, 1, 1.5, 2, 3, 4
   or 6 (OCP Microscaling Formats 1.0, 5.3.3). *)
let e2m1_value c =
  let v = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |].(c land 7) in
  if c land 8 <> 0 then -.v else v

(* float16's bits of 2^e, for e a normal exponent. *)
let half_power e = (e + 15) lsl 10

type weight = Mxfp4 | Q8_0 | Q6_k

(* Each weight by its a's rows: Q8_0 at 3 and at 40, whose tiles step by 32
   terms and by 16. *)
let weights = [ (Mxfp4, 1); (Q8_0, 3); (Q8_0, 40); (Q6_k, 2) ]

let weight_name = function
  | Mxfp4 -> "MXFP4 by bfloat16"
  | Q8_0 -> "Q8_0 by float32"
  | Q6_k -> "Q6_K's int8 by float16"

let weight_id (w, m) = Printf.sprintf "%s, %d rows" (weight_name w) m

(* A weight of [n] rows of [k] terms laid out as nx_quant stores it for a
   product: b, its scale, the shape a's rows take, the value of b's element at
   row j and term t, and a's dtype. MXFP4 is codes packed two a byte, the first
   in the low bits, by Exponent codes a block of 32 terms; Q8_0 blocks of 34
   bytes, a float16 scale then 32 int8 quants; Q6_K int8 quants by float32
   scales a block of 16 terms. *)
let weight_of bk ~n ~k = function
  | Mxfp4 ->
      let code j t = ((5 * j) + (3 * t)) mod 16 in
      let exponent j q = 125 + ((j + q) mod 5) in
      let bytes =
        Array.init
          (n * k / 2)
          (fun i ->
            let j = i / (k / 2) and t = 2 * (i mod (k / 2)) in
            code j t lor (code j (t + 1) lsl 4))
      in
      let packed =
        Support.on bk (A.of_array D.Uint8 (Iarray.of_list [ n; k / 2 ]) bytes)
      in
      let b = Option.get (A.bitcast D.Float4_e2m1fn packed) in
      let b = Option.get (A.move (M.Reshape (Iarray.of_list [ n; k ])) b) in
      let codes =
        Array.init
          (n * k / 32)
          (fun i -> exponent (i / (k / 32)) (i mod (k / 32)))
      in
      let scale =
        Support.on bk (A.of_array D.Uint8 (Iarray.of_list [ n; k / 32 ]) codes)
      in
      let value j t =
        Float.ldexp (e2m1_value (code j t)) (exponent j (t / 32) - 127)
      in
      (A.Any b, Nx_kernel.Exponent scale, [ k ], value, D.Any D.Bfloat16)
  | Q8_0 ->
      let nb = k / 32 in
      let power j q = ((j + q) mod 5) - 2 in
      let quant j q i = (((5 * j) + (3 * q) + i) mod 17) - 8 in
      let byte t =
        let j = t / (nb * 34) and q = t / 34 mod nb and i = t mod 34 in
        match i with
        | 0 -> half_power (power j q) land 255
        | 1 -> half_power (power j q) lsr 8
        | i -> quant j q (i - 2) land 255
      in
      let blocks =
        Support.on bk
          (A.of_array D.Uint8
             (Iarray.of_list [ n; nb; 34 ])
             (Array.init (n * nb * 34) byte))
      in
      let slice from count =
        let r start count = { M.start; count; step = 1 } in
        Option.get
          (A.move
             (M.Slice (Iarray.of_list [ r 0 n; r 0 nb; r from count ]))
             blocks)
      in
      let b = Option.get (A.bitcast D.Int8 (slice 2 32)) in
      let d = Option.get (A.bitcast D.Float16 (slice 0 2)) in
      let d = Option.get (A.move (M.Reshape (Iarray.of_list [ n; nb; 1 ])) d) in
      let value j t =
        Float.ldexp
          (Float.of_int (quant j (t / 32) (t mod 32)))
          (power j (t / 32))
      in
      (A.Any b, Nx_kernel.Value d, [ nb; 32 ], value, D.Any D.Float32)
  | Q6_k ->
      let quant j t = (((3 * j) + t) mod 64) - 32 in
      let power j q = ((j + q) mod 5) - 4 in
      let b =
        Support.on bk
          (A.of_array D.Int8
             (Iarray.of_list [ n; k ])
             (Array.init (n * k) (fun i -> quant (i / k) (i mod k))))
      in
      let d =
        Support.on bk
          (A.of_array D.Float32
             (Iarray.of_list [ n; k / 16 ])
             (Array.init
                (n * k / 16)
                (fun i ->
                  Float.ldexp 1. (power (i / (k / 16)) (i mod (k / 16))))))
      in
      let value j t =
        Float.ldexp (Float.of_int (quant j t)) (power j (t / 16))
      in
      (A.Any b, Nx_kernel.Value d, [ k ], value, D.Any D.Float16)

(* nx_metal.mli: a call's launches read a weight and its scale where they lie:
   loaders only, no scale pass, copy or widening. *)
let read_in_place launches =
  check_loaders ~keyed:true launches;
  equal (list string) [] (scale_passes launches);
  let moves =
    List.filter_map
      (fun (t : Plan.traced) ->
        match t.kernel with
        | Nx_metal.Kernels.Copy | Copy_8 | Copy_16 | Copy_32 | Copy_64 | Widen
          ->
            Some (Nx_metal.Kernels.name t.kernel)
        | _ -> None)
      launches
  in
  equal ~msg:"copies and widenings" (list string) [] moves

(* nx_metal.mli: a weight as a model stores it takes loaders, which read it and
   its scales where they lie: no copy, widening or scale pass, as a, contiguous,
   and dst, of a's dtype, take none either. Each output is the sum of a's
   elements times the weight's values, exact here in float32, rounded once into
   dst. *)
let test_weight (w, m) =
  let bk = backend () in
  let n = 40 and k = 64 in
  let b, scale, inner, value, adt = weight_of bk ~n ~k w in
  let a_at r t = Float.of_int (((r + t) mod 7) - 3) in
  let a_values = Array.init (m * k) (fun i -> a_at (i / k) (i mod k)) in
  let a =
    Contract_cases.on bk
      (Contract_cases.cast_to adt
         (A.Any (A.of_array D.Float64 (Iarray.of_list (m :: inner)) a_values)))
  in
  let p = List.length inner in
  let contracting = Iarray.of_list (List.init p (fun i -> (1 + i, 1 + i))) in
  let (D.Any ddt) = adt in
  let dst = A.create (S.device ()) ddt (Iarray.of_list [ m; n ]) in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract ~batch:(Iarray.of_list []) ~contracting ~segments:None
      ~acc:(D.Any D.Float32) ~dst ~init:None
      { Nx_kernel.array = a; scale = None }
      { Nx_kernel.array = b; scale = Some scale }
  in
  equal answer A.Done got;
  read_in_place launches;
  let sums =
    Array.init (m * n) (fun o ->
        let r = o / n and j = o mod n in
        let s = ref 0. in
        for t = 0 to k - 1 do
          s := !s +. (a_at r t *. value j t)
        done;
        !s)
  in
  let as64 x =
    A.to_array (A.expect D.Float64 (Contract_cases.cast_to (D.Any D.Float64) x))
  in
  let want =
    as64
      (Contract_cases.cast_to adt
         (A.Any (A.of_array D.Float64 (Iarray.of_list [ m; n ]) sums)))
  in
  equal (array float_exact) want (as64 (A.Any (Support.host dst)))

(* The Exponent codes at the bounds of e2m1's exact product (2 and 252), past
   them (0, 1, 253, 254 and NaN's 255) and inside (127), and e2m1's codes of +0,
   -0, +-0.5 and +-6, the exponent field's extremes. *)
let edge_codes = [ 0; 1; 2; 127; 252; 253; 254; 255 ]
let edge_elements = [ 0; 8; 1; 9; 7; 15 ]

(* nx_metal.mli: an MXFP4 weight's loader takes each element times its code's
   power of two by the arithmetic of its bits. Column j of a weight of 64 terms
   holds one pair of an element and a code: the element at term 0 (every term
   for -0), the code for terms 0-31 and 127 for the rest, the other terms +0.
   With a of ones, each output is the element times its code, then +0s, as one
   rounded sum. *)
let test_exponent_edges () =
  let bk = backend () in
  let pairs =
    Array.of_list
      (List.concat_map
         (fun e -> List.map (fun c -> (c, e)) edge_elements)
         edge_codes)
  in
  let n = Array.length pairs and k = 64 and m = 64 in
  let element j t =
    let c, _ = pairs.(j) in
    if t = 0 || c = 8 then c else 0
  in
  let bytes =
    Array.init
      (n * k / 2)
      (fun i ->
        let j = i / (k / 2) and t = 2 * (i mod (k / 2)) in
        element j t lor (element j (t + 1) lsl 4))
  in
  let packed =
    Support.on bk (A.of_array D.Uint8 (Iarray.of_list [ n; k / 2 ]) bytes)
  in
  let b = Option.get (A.bitcast D.Float4_e2m1fn packed) in
  let b = Option.get (A.move (M.Reshape (Iarray.of_list [ n; k ])) b) in
  let codes =
    Array.init (n * 2) (fun i -> if i mod 2 = 0 then snd pairs.(i / 2) else 127)
  in
  let scale =
    Support.on bk (A.of_array D.Uint8 (Iarray.of_list [ n; 2 ]) codes)
  in
  let a =
    Support.on bk
      (A.of_array D.Bfloat16 (Iarray.of_list [ m; k ]) (Array.make (m * k) 1.))
  in
  let dst = A.create (S.device ()) D.Float32 (Iarray.of_list [ m; n ]) in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract ~batch:(Iarray.of_list [])
      ~contracting:(Iarray.of_list [ (1, 1) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst ~init:None
      { Nx_kernel.array = A.Any a; scale = None }
      { Nx_kernel.array = A.Any b; scale = Some (Nx_kernel.Exponent scale) }
  in
  equal answer A.Done got;
  read_in_place launches;
  let column j =
    let c, e = pairs.(j) in
    let rest = if e = 255 then Float.nan else 0. in
    exponent_scaled (e2m1_value c) e +. rest
  in
  let want = Array.init (m * n) (fun o -> column (o mod n)) in
  equal (array float_exact) want (A.to_array (Support.host dst))

(* [x] with its leading axis split into [g] of the rest. *)
let split_lead g (A.Any x) =
  match Iarray.to_list (A.Layout.shape (A.layout x)) with
  | e :: rest ->
      A.Any
        (Option.get
           (A.move (M.Reshape (Iarray.of_list (g :: (e / g) :: rest))) x))
  | [] -> invalid_arg "split_lead"

let split_scale g : Nx_kernel.scale -> Nx_kernel.scale = function
  | Exponent e -> Exponent (A.expect D.Uint8 (split_lead g (A.Any e)))
  | Value v ->
      let (A.Any v) = split_lead g (A.Any v) in
      Value v

(* Each weight of experts as nx_quant's experts product reads it. *)
let experts = [ Mxfp4; Q8_0 ]

(* nx_metal.mli: experts of a weight as a model stores it, [g] of [n] rows, in a
   Rows call, as nx_quant's: loaders read the weight and its scales where they
   lie, no copy, widening or scale pass; a's rows, contiguous, and dst need
   none. Row r's output is its expert ids.(r)'s rows times a's row, exact here
   in float32, rounded once into dst. *)
let experts_with ~n check w =
  let bk = backend () in
  let g = 3 and k = 64 in
  let ids = [| 2; 0; 1; 2; 0 |] in
  let p = Array.length ids in
  let b, scale, inner, value, adt = weight_of bk ~n:(g * n) ~k w in
  let b = split_lead g b and scale = split_scale g scale in
  let a_at r t = Float.of_int (((r + t) mod 7) - 3) in
  let a_values = Array.init (p * k) (fun i -> a_at (i / k) (i mod k)) in
  let a =
    Contract_cases.on bk
      (Contract_cases.cast_to adt
         (A.Any (A.of_array D.Float64 (Iarray.of_list (p :: inner)) a_values)))
  in
  let q = List.length inner in
  let contracting = Iarray.of_list (List.init q (fun i -> (1 + i, 2 + i))) in
  let ids_dev =
    Support.on bk
      (A.of_array D.Int64 (Iarray.of_list [ p ]) (Array.map Int64.of_int ids))
  in
  let (D.Any ddt) = adt in
  let dst = A.create (S.device ()) ddt (Iarray.of_list [ p; n ]) in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract ~batch:(Iarray.of_list []) ~contracting
      ~segments:
        (Some (Nx_kernel.Rows { ids = ids_dev; a_axis = 0; b_axis = 0 }))
      ~acc:(D.Any D.Float32) ~dst ~init:None
      { Nx_kernel.array = a; scale = None }
      { Nx_kernel.array = b; scale = Some scale }
  in
  equal answer A.Done got;
  check launches;
  let sums =
    Array.init (p * n) (fun o ->
        let r = o / n and j = o mod n in
        let s = ref 0. in
        for t = 0 to k - 1 do
          s := !s +. (a_at r t *. value ((ids.(r) * n) + j) t)
        done;
        !s)
  in
  let as64 x =
    A.to_array (A.expect D.Float64 (Contract_cases.cast_to (D.Any D.Float64) x))
  in
  let want =
    as64
      (Contract_cases.cast_to adt
         (A.Any (A.of_array D.Float64 (Iarray.of_list [ p; n ]) sums)))
  in
  equal (array float_exact) want (as64 (A.Any (Support.host dst)))

let test_experts w = experts_with ~n:40 read_in_place w

(* nx_metal.mli: the same experts at 256 columns or more, a's 5 rows within the
   keys' bound of 16, take the vector kernels, which read the weight and its
   scales where they lie, Q8_0's blocks included: one launch, and no copy,
   widening or scale pass. *)
let vector_in_place launches =
  let names p =
    List.filter_map
      (fun (t : Plan.traced) ->
        if p t.kernel then Some (Nx_metal.Kernels.name t.kernel) else None)
      launches
  in
  equal ~msg:"the vector kernels' launches" int 1
    (List.length (names (tile_prefix "contract_mv_")));
  equal ~msg:"copies, widenings and scale passes" (list string) []
    (names (fun k -> apart_copy k || k = Widen || k = Contract_scale))

let test_vector_experts w = experts_with ~n:300 vector_in_place w

(* An int8 weight of blocks of [bl] terms, as Q8_0's of 32, [gap] elements
   apart, its first block [at] past its buffer's start; [g] groups of a Rows
   call, or none and [bt] batch elements; a's [rows], of [adt]; b's [n] columns;
   [nb] blocks; scales of [sdt], one a block or one an element. *)
type bcase = {
  bl : int;
  nb : int;
  gap : int;
  at : int;
  bg : int;
  bbt : int;
  brows : int;
  bn : int;
  badt : D.any;
  bsdt : D.any;
  per_block : bool;
  bseed : int;
}

let pp_bcase ppf c =
  let (D.Any a) = c.badt in
  let (D.Any sd) = c.bsdt in
  Format.fprintf ppf
    "%d blocks of %d terms %d apart, at %d, %s, %d rows of %s, %d columns, \
     scales of %s %s, seed %d"
    c.nb c.bl c.gap c.at
    (if c.bg > 0 then Printf.sprintf "%d groups" c.bg
     else Printf.sprintf "batch %d" c.bbt)
    c.brows (D.name a) c.bn (D.name sd)
    (if c.per_block then "a block" else "an element")
    c.bseed

(* Blocks of 24 and 48 terms, so that tiles stepping by 16 or 32 terms step
   across a block, at a's rows that take each, in Rows calls and unsegmented
   ones. Only blocks of 24 put a step of 16 across one, so they are drawn twice
   as often. *)
let bcase =
  Gen.(
    with_pp pp_bcase
      (let+ bl = frequency [ (2, constant 24); (1, constant 48) ]
       and+ nb = int_range 2 4
       and+ gap = int_range 1 3
       and+ at_gap = bool
       and+ bg = frequency [ (1, constant 0); (1, int_range 2 3) ]
       and+ bbt = int_range 1 2
       and+ brows = frequency [ (1, int_range 1 6); (1, int_range 33 48) ]
       and+ bn = int_range 17 48
       and+ badt = of_list D.[ Any Float32; Any Float16; Any Bfloat16 ]
       and+ bsdt = of_list D.[ Any Float16; Any Float32 ]
       and+ per_block = bool
       and+ bseed = int_range 0 1000 in
       {
         bl;
         nb;
         gap;
         at = (if at_gap then gap else 0);
         bg;
         bbt;
         brows;
         bn;
         badt;
         bsdt;
         per_block;
         bseed;
       }))

(* Steps of 32 and of 16 across blocks of 24, unsegmented and in Rows calls, and
   a step of 16 within blocks of 48. *)
let blocks_examples =
  let c =
    {
      bl = 24;
      nb = 3;
      gap = 2;
      at = 2;
      bg = 0;
      bbt = 1;
      brows = 3;
      bn = 40;
      badt = D.Any D.Float32;
      bsdt = D.Any D.Float16;
      per_block = true;
      bseed = 0;
    }
  in
  [
    c;
    { c with brows = 40 };
    { c with bl = 48; brows = 40 };
    { c with bg = 3; brows = 5 };
    { c with bg = 2; brows = 40; per_block = false };
    { c with bg = 2; bl = 48; brows = 40; per_block = false };
  ]

(* nx_metal.mli: a weight whose terms make two runs, the inner contiguous, is
   read where it lies, its scales too: loaders only, no copy, widening or scale
   pass, in a Rows call or not. Each output is a's row times its weight's row,
   exact here in float32, rounded once into dst, wherever a tile's step of terms
   falls across a block. *)
let law_blocks c =
  let bk = backend () in
  let k = c.nb * c.bl in
  let lead = if c.bg > 0 then c.bg else c.bbt in
  let rows_b = lead * c.bn in
  let quant j t = (((5 * j) + (3 * t) + c.bseed) mod 17) - 8 in
  let power j q = ((j + q + c.bseed) mod 5) - 2 in
  let scale_at j t = power j (if c.per_block then t / c.bl else t) in
  let value j t = Float.ldexp (Float.of_int (quant j t)) (scale_at j t) in
  (* b's buffer: [lead; n; nb; bl + gap], each block's quants at [at, at + bl),
     127 elsewhere. *)
  let stride = c.bl + c.gap in
  let held =
    Array.init
      (rows_b * c.nb * stride)
      (fun e ->
        let j = e / (c.nb * stride)
        and q = e / stride mod c.nb
        and i = e mod stride in
        if i < c.at || i >= c.at + c.bl then 127
        else quant j ((q * c.bl) + i - c.at))
  in
  let buffer =
    Support.on bk
      (A.of_array D.Int8 (Iarray.of_list [ lead; c.bn; c.nb; stride ]) held)
  in
  let r start count = { M.start; count; step = 1 } in
  let b =
    Option.get
      (A.move
         (M.Slice (Iarray.of_list [ r 0 lead; r 0 c.bn; r 0 c.nb; r c.at c.bl ]))
         buffer)
  in
  let inner = if c.per_block then 1 else c.bl in
  let scales =
    Array.init
      (rows_b * c.nb * inner)
      (fun e ->
        let j = e / (c.nb * inner)
        and q = e / inner mod c.nb
        and i = e mod inner in
        Float.ldexp 1. (scale_at j ((q * c.bl) + i)))
  in
  let (A.Any sv) =
    Contract_cases.on bk
      (Contract_cases.cast_to c.bsdt
         (A.Any
            (A.of_array D.Float64
               (Iarray.of_list [ lead; c.bn; c.nb; inner ])
               scales)))
  in
  let a_at row t = Float.of_int (((row + t + c.bseed) mod 7) - 3) in
  let a_rows = if c.bg > 0 then c.brows else c.bbt * c.brows in
  let a_shape =
    if c.bg > 0 then [ c.brows; c.nb; c.bl ] else [ c.bbt; c.brows; c.nb; c.bl ]
  in
  let a =
    Contract_cases.on bk
      (Contract_cases.cast_to c.badt
         (A.Any
            (A.of_array D.Float64 (Iarray.of_list a_shape)
               (Array.init (a_rows * k) (fun e -> a_at (e / k) (e mod k))))))
  in
  let ids =
    Array.init c.brows (fun r -> ((r * 7) + c.bseed) mod Int.max 1 c.bg)
  in
  let (D.Any ddt) = c.badt in
  let d_shape =
    if c.bg > 0 then [ c.brows; c.bn ] else [ c.bbt; c.brows; c.bn ]
  in
  let dst = A.create (S.device ()) ddt (Iarray.of_list d_shape) in
  cover "a Rows call" (c.bg > 0);
  cover "an unsegmented call" (c.bg = 0);
  let got, launches =
    Plan.traced @@ fun () ->
    if c.bg > 0 then
      let ids_dev =
        Support.on bk
          (A.of_array D.Int64
             (Iarray.of_list [ c.brows ])
             (Array.map Int64.of_int ids))
      in
      Nx_metal.contract ~batch:(Iarray.of_list [])
        ~contracting:(Iarray.of_list [ (1, 2); (2, 3) ])
        ~segments:
          (Some (Nx_kernel.Rows { ids = ids_dev; a_axis = 0; b_axis = 0 }))
        ~acc:(D.Any D.Float32) ~dst ~init:None
        { Nx_kernel.array = a; scale = None }
        { Nx_kernel.array = A.Any b; scale = Some (Value sv) }
    else
      Nx_metal.contract
        ~batch:(Iarray.of_list [ (0, 0) ])
        ~contracting:(Iarray.of_list [ (2, 2); (3, 3) ])
        ~segments:None ~acc:(D.Any D.Float32) ~dst ~init:None
        { Nx_kernel.array = a; scale = None }
        { Nx_kernel.array = A.Any b; scale = Some (Value sv) }
  in
  equal answer A.Done got;
  read_in_place launches;
  (* The launched loader's step of terms, its tile's depth at float32: a
     loader's name ends in its tile's size and b's form. *)
  let step =
    List.find_map
      (fun (t : Plan.traced) ->
        if not (is_loader t.kernel) then None
        else
          match
            List.rev (String.split_on_char '_' (Nx_metal.Kernels.name t.kernel))
          with
          | _ :: "32" :: _ -> Some (C.depth C.F32 C.T32)
          | _ :: "16" :: _ -> Some (C.depth C.F32 C.T16)
          | _ -> None)
      launches
  in
  let across s =
    List.exists
      (fun q -> q * c.bl mod s <> 0)
      (List.init (c.nb - 1) (fun q -> q + 1))
  in
  cover "a step of 16 terms across a block" (step = Some 16 && across 16);
  cover "a step of 32 terms across a block" (step = Some 32 && across 32);
  cover "a Rows call's step of 16 across a block"
    (c.bg > 0 && step = Some 16 && across 16);
  cover "a Rows call's step of 32 across a block"
    (c.bg > 0 && step = Some 32 && across 32);
  let sums =
    Array.init (a_rows * c.bn) (fun o ->
        let row = o / c.bn and col = o mod c.bn in
        let j =
          if c.bg > 0 then (ids.(row) * c.bn) + col
          else (row / c.brows * c.bn) + col
        in
        let s = ref 0. in
        for t = 0 to k - 1 do
          s := !s +. (a_at row t *. value j t)
        done;
        !s)
  in
  let as64 x =
    A.to_array (A.expect D.Float64 (Contract_cases.cast_to (D.Any D.Float64) x))
  in
  let want =
    as64
      (Contract_cases.cast_to c.badt
         (A.Any (A.of_array D.Float64 (Iarray.of_list [ a_rows * c.bn ]) sums)))
  in
  equal (array float_exact) want (as64 (A.Any (Support.host dst)))

(* The vector kernels *)

(* nx_metal.mli: the wider operand's keys of the vector kernels, and each key's
   bound on a call's rows: 16 for float4_e2m1fn by Exponent codes and int8 by
   Values of float16 or float32, 2 for float16 and bfloat16, 1 for float32; and
   the least columns, 256. *)
type vkey = Vf32 | Vf16 | Vbf16 | Ve2m1 | Vi8

let vkey_name = function
  | Vf32 -> "float32"
  | Vf16 -> "float16"
  | Vbf16 -> "bfloat16"
  | Ve2m1 -> "float4_e2m1fn by Exponent"
  | Vi8 -> "int8 by Value"

(* nx_metal.mli: a key's bound on a call's rows, a Rows call's where [rows]. *)
let vkey_bound ~rows = function
  | (Ve2m1 | Vi8) when rows -> 16
  | Ve2m1 -> 11
  | Vi8 -> 3
  | Vf16 | Vbf16 -> 2
  | Vf32 -> 1

(* nx_metal.mli: a key's bound on a call's terms, a Rows call's where [rows]. *)
let vkey_terms ~rows = function
  | Vf32 -> 1 lsl 23
  | Vf16 | Vbf16 -> 1 lsl 22
  | Ve2m1 | Vi8 -> if rows then 1 lsl 18 else 1 lsl 19

let vector_columns = 256

let vkey_of (o : Nx_kernel.operand) =
  let (A.Any x) = o.array in
  match (A.dtype x, o.scale) with
  | D.Float32, None -> Some Vf32
  | D.Float16, None -> Some Vf16
  | D.Bfloat16, None -> Some Vbf16
  | D.Float4_e2m1fn, Some (Exponent _) -> Some Ve2m1
  | D.Int8, Some (Value v) -> (
      match A.dtype v with D.Float16 | D.Float32 -> Some Vi8 | _ -> None)
  | _ -> None

(* nx_metal.mli: the other operand is of float32, float16 or bfloat16, or
   scaled. *)
let vector_side (o : Nx_kernel.operand) =
  let (A.Any x) = o.array in
  Option.is_some o.scale || List.mem (D.Any (A.dtype x)) native

(* nx_metal.mli: whether a float32 call of terms, of [m] rows and [n] columns a
   batch element, takes the vector kernels, [Some true] where exchanged: b of a
   key, its rows at most the key's bound and its columns at least 256; or, but
   in a Rows call, a of a key, its columns at most the key's bound and its rows
   at least 256. *)
let vector_rule ?(terms = max_int) ~rows ~m ~n ~k oa ob =
  let within key = k <= Int.min terms (vkey_terms ~rows key) in
  if k = 0 then None
  else
    match vkey_of ob with
    | Some kb
      when vector_side oa
           && m <= vkey_bound ~rows kb
           && n >= vector_columns && within kb ->
        Some false
    | _ -> (
        match vkey_of oa with
        | Some ka
          when (not rows) && vector_side ob
               && n <= vkey_bound ~rows ka
               && m >= vector_columns && within ka ->
            Some true
        | _ -> None)

(* nx_metal.mli's lanes of an output of [r] spans: the most power of two at most
   [r] and 32. *)
let vector_lanes r =
  let rec up s = if 2 * s <= r && s < 32 then up (2 * s) else s in
  if r <= 0 then 0 else up 1

(* nx_metal.mli's vector association: span j of r = ⌈k / 32⌉ to lane j mod S,
   each lane's terms summed from +0 by fused multiply-adds in increasing k; for
   h = S/2, ..., 1, lane l < h adds lane l + h's sum to its own; then init. An
   output of no product is init, or +0. Of the terms [(a.(a0 + kk), b.(b0 +
   kk))]: the S spans that start at a multiple of S, one a lane, summed
   together, each lane's terms in increasing k, so that the processor overlaps
   the lanes' sums. *)
let vectored_at ~k ~init a a0 b b0 =
  let r = (k + 31) / 32 in
  if k = 0 then Option.value init ~default:0.
  else begin
    let s = vector_lanes r in
    let lanes = Array.make s 0. in
    for g = 0 to ((r + s - 1) / s) - 1 do
      for t = 0 to 31 do
        for l = 0 to s - 1 do
          let kk = (32 * ((g * s) + l)) + t in
          if kk < k then
            lanes.(l) <-
              fma32
                (Array.unsafe_get a (a0 + kk))
                (Array.unsafe_get b (b0 + kk))
                lanes.(l)
        done
      done
    done;
    let h = ref (s / 2) in
    while !h >= 1 do
      for l = 0 to !h - 1 do
        lanes.(l) <- add32 lanes.(l) lanes.(l + !h)
      done;
      h := !h / 2
    done;
    match init with None -> lanes.(0) | Some i -> add32 lanes.(0) i
  end

let vector_kernel k = tile_prefix "contract_mv_" k

(* kernels.h's NX_MV_FORM of each n x k vector launch: 2 loads a lane's whole
   spans by 16-byte words, 1 by 2-byte words, 3 by 16-byte words with two scale
   blocks a span, 0 each term where it lies. *)
let vector_forms launches =
  List.filter_map
    (fun (t : Plan.traced) ->
      let name = Nx_metal.Kernels.name t.kernel in
      if
        vector_kernel t.kernel
        && List.exists
             (fun suffix -> String.ends_with ~suffix name)
             [ "_nk"; "_nk_typed"; "_nk_w32"; "_nk_typed_w32" ]
      then Some (word t Nx_metal.Kernels.mv_form)
      else None)
    launches

(* contract.mli: a vector plan's lanes, S, of an output's spans, in either
   form. *)
let vector_lane_rows =
  [
    (1, 1);
    (32, 1);
    (33, 2);
    (64, 2);
    (992, 16);
    (1024, 32);
    (1056, 32);
    (2880, 32);
  ]

let test_vector_lanes (k, want) =
  let e =
    {
      C.batch = 1;
      m = 1;
      n = 4096;
      k;
      stage = C.F32;
      nk = true;
      complex = false;
      source = 0;
    }
  in
  List.iter
    (fun kn ->
      match C.Vector.plan e ~key:C.Vector.Dense_f16 ~rows:false ~kn with
      | None -> fail "no plan"
      | Some p ->
          equal ~msg:"the lanes" int want p.lanes;
          equal ~msg:"the spans" int (cdiv k 32) p.spans)
    [ false; true ]

(* A call the vector kernels may take: the wider operand's key [xkey], its
   scale's blocks along k [xblock] (32, 1 or one block, 0), an int8's scale
   dtype [xsd]; the other operand's dtype [xvec], and whether it is scaled;
   exchanged, or a Rows call of [xgroups] groups; extents; the wider operand
   terms or columns contiguous ([xkn]); both operands at an odd offset, or the
   wider [xwskip] elements in; an init and dst; a Rows call's a of rows of two
   axes grouped along m ([xpair]). *)
type pair = No_pair | Pair_after | Pair_before

type vcall = {
  xkey : vkey;
  xblock : int;
  xrn : int;
  xsd : D.any;
  xvec : D.any;
  xscaled : bool;
  xexch : bool;
  xgroups : int option;
  xids : drawn_ids;
  xbt : int;
  xm : int;
  xn : int;
  xk : int;
  xkn : bool;
  xoff : bool;
  xwskip : int;
  xinit : bool;
  xdst : D.any;
  xsplit : bool;
  xpair : pair;
  xterms : int option;
      (** {!C.Vector.bound_terms} at [xk] plus this, 0 or -1: the call's terms
          at the bound or one past. *)
  xseed : int;
}

let pp_vcall ppf c =
  let name (D.Any d) = D.name d in
  Format.fprintf ppf
    "%d x %s%d%s x %d x %d, %s%s%s%s by %s%s%s%s%s, %sdst %s%s%s, seed %d" c.xbt
    (if c.xpair = Pair_before then "2 x " else "")
    c.xm
    (if c.xpair = Pair_after then " x 2" else "")
    c.xn c.xk (vkey_name c.xkey)
    (if c.xkey = Vi8 then " of " ^ name c.xsd else "")
    ((match (c.xkey, c.xblock) with
       | (Ve2m1 | Vi8), 0 -> ", a scale a column"
       | (Ve2m1 | Vi8), b -> Printf.sprintf ", scale blocks of %d" b
       | _ -> "")
    ^
    match c.xkey with
    | (Ve2m1 | Vi8) when c.xrn > 1 -> " and of 2 columns"
    | _ -> "")
    (if c.xkn then ", columns contiguous" else "")
    (name c.xvec)
    (if c.xscaled then " scaled" else "")
    (if c.xexch then ", exchanged" else "")
    (match c.xgroups with
    | Some g -> Printf.sprintf ", Rows of %d groups, ids %s" g (ids_name c.xids)
    | None -> "")
    (if c.xoff then Printf.sprintf ", operands %d elements in" skipped
     else if c.xwskip > 0 then
       Printf.sprintf ", the wider %d elements in" c.xwskip
     else "")
    (if c.xinit then "init, " else "")
    (name c.xdst)
    (if c.xsplit then ", launches under 0.6 of its estimate" else "")
    (match c.xterms with
    | Some d -> Printf.sprintf ", terms bound to %d" (Int.max 1 (c.xk + d))
    | None -> "")
    c.xseed

(* The most terms of all outputs a draw takes, so that the host's reference
   stays short. *)
let vector_terms = 1 lsl 22

(* Rows about each key's bound and columns about 256; k about the lanes' counts:
   spans of 1, 2, 31, 32, 33 and 90, partial ones too. *)
let vcall_case =
  let open Gen in
  with_pp pp_vcall
    (let+ xkey = of_list [ Vf32; Vf16; Vbf16; Ve2m1; Vi8 ]
     and+ xblock =
       frequency
         [
           (4, constant 32);
           (2, of_list [ 8; 16; 24; 48 ]);
           (1, constant 1);
           (1, constant 0);
         ]
     and+ xrn = frequency [ (3, constant 1); (1, constant 2) ]
     and+ xsd = of_list D.[ Any Float16; Any Float32 ]
     and+ xvec = of_list native
     and+ xscaled = frequency [ (5, constant false); (1, constant true) ]
     and+ xexch = frequency [ (4, constant false); (1, constant true) ]
     and+ rows =
       frequency [ (3, constant None); (1, map Option.some (int_range 1 4)) ]
     and+ xids = of_list [ Each; One; Outside ]
     and+ xbt = frequency [ (3, constant 1); (1, constant 2) ]
     and+ near =
       frequency
         [ (2, constant 0); (2, constant 1); (1, constant 2); (1, constant 3) ]
     and+ wide = of_list [ 255; 256; 257; 300; 520 ]
     and+ xk =
       frequency
         [
           (1, of_list [ 1; 31; 33 ]);
           (2, of_list [ 32; 64; 96; 100 ]);
           (2, of_list [ 992; 1024; 1056 ]);
           (1, constant 2880);
         ]
     and+ xkn = frequency [ (3, constant false); (1, constant true) ]
     and+ xoff = frequency [ (3, constant false); (1, constant true) ]
     and+ xinit = bool
     and+ xdst =
       frequency [ (3, constant (D.Any D.Float32)); (1, of_list native) ]
     and+ pair =
       frequency
         [
           (6, constant No_pair);
           (1, constant Pair_after);
           (1, constant Pair_before);
         ]
     and+ xterms =
       frequency
         [
           (6, constant None); (1, constant (Some 0)); (1, constant (Some (-1)));
         ]
     and+ xseed = nat in
     (* The key's bound and one past it, else fewer rows. *)
     let xgroups = if xexch then None else rows in
     let bound = vkey_bound ~rows:(Option.is_some xgroups) xkey in
     let few =
       match near with
       | 0 -> 1
       | 1 -> bound
       | 2 -> bound + 1
       | _ -> 1 + (bound / 2)
     in
     let xpair = if Option.is_some xgroups then pair else No_pair in
     let xm, xn = if xexch then (wide, few) else (few, wide) in
     let xm = if xpair <> No_pair then Int.max 1 (xm / 2) else xm in
     let q = if xpair <> No_pair then 2 else 1 in
     let xk = Int.max 1 (Int.min xk (vector_terms / (xbt * xm * q * xn))) in
     {
       xkey;
       xblock;
       xrn;
       xsd;
       xvec;
       xscaled;
       xexch;
       xgroups;
       xids;
       xbt;
       xm;
       xn;
       xk;
       xkn;
       xoff;
       xwskip = 0;
       xinit;
       xdst;
       xsplit = false;
       xpair;
       xterms;
       xseed;
     })

(* The wider operand of [c], of logical [shape] whose last two axes are its
   columns and terms, laid out with its terms contiguous or, where [c.xkn], its
   columns: the operand, its scale, and its element's value at an index of
   [shape], scaled. *)
let vector_matrix st bk c shape =
  let rank = List.length shape in
  let stored = if c.xkn then swap_last shape else shape in
  let view =
    if c.xoff then `Skip skipped
    else if c.xwskip > 0 then `Skip c.xwskip
    else `Fresh
  in
  let dt =
    match c.xkey with
    | Vf32 -> D.Any D.Float32
    | Vf16 -> D.Any D.Float16
    | Vbf16 -> D.Any D.Bfloat16
    | Ve2m1 -> D.Any D.Float4_e2m1fn
    | Vi8 -> D.Any D.Int8
  in
  let special = Int.max 64 (4 * c.xk) in
  let x, v = laid_operand ~special st bk dt stored ~low:(-8) view in
  let x =
    if not c.xkn then x
    else
      let (A.Any x) = x in
      A.Any
        (Option.get
           (A.move
              (M.Permute (Iarray.of_list (swap_last (List.init rank Fun.id))))
              x))
  in
  let raw idx = v.(number_of stored (if c.xkn then swap_last idx else idx)) in
  (* The scale's blocks along k: of 32 terms where k holds them ([xblock] 32),
     else of 2 or 1; of 8, 16, 24 or 48 where k holds them, else of 1; of one
     term (1); one block (0). Along the columns, of one ([xrn] 1), else of 32, 2
     or 1 as they hold. *)
  let along = match c.xblock with 0 -> 1 | 1 -> 0 | 32 -> 2 | b -> b in
  let across = if c.xrn > 1 then 2 else 0 in
  let ratios =
    List.init rank (fun i ->
        if i = rank - 1 then along else if i = rank - 2 then across else 0)
  in
  let spec exp sdt =
    { sexp = exp; sdt; sratios = ratios; sview = Fresh_scale }
  in
  match c.xkey with
  | Vf32 | Vf16 | Vbf16 -> (x, None, raw)
  | Ve2m1 ->
      let s, f = drawn_scale st bk (spec true (D.Any D.Uint8)) shape in
      (x, Some s, fun idx -> f idx (raw idx))
  | Vi8 ->
      let s, f = drawn_scale st bk (spec false c.xsd) shape in
      (x, Some s, fun idx -> f idx (raw idx))

(* The other operand of [c], of logical [shape], its terms last: the operand,
   its scale where [c.xscaled], a Value of float32 a block of 32 terms, and its
   value at an index, scaled. *)
let vector_operand st bk c shape ~kn =
  let rank = List.length shape in
  let stored = if kn then swap_last shape else shape in
  let view = if c.xoff then `Skip skipped else `Fresh in
  let special = Int.max 64 (4 * c.xk) in
  let x, v = laid_operand ~special st bk c.xvec stored ~low:(-8) view in
  let x =
    if not kn then x
    else
      let (A.Any x) = x in
      A.Any
        (Option.get
           (A.move
              (M.Permute (Iarray.of_list (swap_last (List.init rank Fun.id))))
              x))
  in
  let raw idx = v.(number_of stored (if kn then swap_last idx else idx)) in
  if not c.xscaled then (x, None, raw)
  else
    let ratios = List.init rank (fun i -> if i = rank - 1 then 2 else 0) in
    let s, f =
      drawn_scale st bk
        {
          sexp = false;
          sdt = D.Any D.Float32;
          sratios = ratios;
          sview = Fresh_scale;
        }
        shape
    in
    (x, Some s, fun idx -> f idx (raw idx))

(* nx_metal.mli: a float32 call whose wider operand is of a key and
   whose rows and terms are at most the key's bounds, or columns
   exchanged, takes
   the vector kernels, one launch of a contract_mv_ function, and every
   output has the bits of the vector association; any other call keeps
   the tiles' association. A Rows call's row of no group is init, or
   +0. Whatever the operands' layouts, a scale's blocks, the dtypes of
   init and dst. *)
(* Of a float4_e2m1fn wider operand [w] by Exponent codes [sw]: whether
   it holds a code of ±0.5, which the vector kernels widen from a
   subnormal half, and codes on both sides of [2, 240], within which a
   span takes their power of two as a float32, outside it each term's
   bits. *)
let e2m1_reach c w sw =
  if c.xkey <> Ve2m1 then (false, false)
  else
    let halves =
      Array.exists
        (fun v -> Float.abs v = 0.5)
        (A.to_array (Contract_cases.to64 w))
    in
    let codes =
      match sw with
      | Some (Nx_kernel.Exponent e) ->
          A.to_array (Contract_cases.to64 (A.Any e))
      | _ -> [||]
    in
    let exact e = e >= 2. && e <= 240. in
    ( halves,
      Array.exists exact codes && Array.exists (fun e -> not (exact e)) codes )

(* Of a float4_e2m1fn weight of [n] columns by Exponent codes [sw] of a code a
   column: whether a k x n lane's 8 columns of one scale block hold a code
   outside [2, 240], where its run takes contract_exponent a column; whether
   such a lane mixes both sides; whether the last, partial, lane holds one. *)
let kn_codes c ~n sw =
  match (c.xkey, sw) with
  | Ve2m1, Some (Nx_kernel.Exponent e) when c.xrn = 1 ->
      let codes = A.to_array (Contract_cases.to64 (A.Any e)) in
      let dims = Iarray.to_list (A.Layout.shape (A.layout e)) in
      let rank = List.length dims in
      let cols = List.nth dims (rank - 2)
      and blocks = List.nth dims (rank - 1) in
      let outside v = v < 2. || v > 240. in
      let general = ref false and mixed = ref false and last = ref false in
      for o = 0 to (Array.length codes / Int.max 1 (cols * blocks)) - 1 do
        for q = 0 to blocks - 1 do
          for g = 0 to (cols - 1) / 8 do
            let lane =
              List.init
                (Int.min 8 (cols - (8 * g)))
                (fun j -> codes.((((o * cols) + (8 * g) + j) * blocks) + q))
            in
            let out = List.exists outside lane in
            if out then general := true;
            if out && List.exists (fun v -> not (outside v)) lane then
              mixed := true;
            if out && cols = n && n mod 8 <> 0 && g = (cols - 1) / 8 then
              last := true
          done
        done
      done;
      (!general, !mixed, !last)
  | _ -> (false, false, false)

let law_vector_bits c =
  let bk = backend () in
  let st = Random.State.make [| c.xseed |] in
  let bt = c.xbt and m = c.xm and n = c.xn and k = c.xk in
  let g = Option.value c.xgroups ~default:0 in
  let segmented = Option.is_some c.xgroups in
  (* a [bt; m; k], [bt; m; 2; k] or [bt; 2; m; k] by [xpair], b [bt; n; k] or,
     in a Rows call, [bt; g; n; k]; the wider operand b, or a where exchanged.
     The call's [m q] rows; a's segmented axis, m. *)
  let q = if c.xpair = No_pair then 1 else 2 in
  let rows = m * q in
  let a_shape, a_axis =
    match c.xpair with
    | No_pair -> ([ bt; m; k ], 1)
    | Pair_after -> ([ bt; m; 2; k ], 1)
    | Pair_before -> ([ bt; 2; m; k ], 2)
  in
  let b_shape = if segmented then [ bt; g; n; k ] else [ bt; n; k ] in
  let a, sa, av, b, sb, bv =
    if c.xexch then
      let a, sa, av = vector_matrix st bk c [ bt; m; k ] in
      let b, sb, bv = vector_operand st bk c b_shape ~kn:c.xkn in
      (a, sa, av, b, sb, bv)
    else
      let a, sa, av = vector_operand st bk c a_shape ~kn:false in
      let b, sb, bv = vector_matrix st bk c b_shape in
      (a, sa, av, b, sb, bv)
  in
  let ids =
    if segmented then ids_drawn st c.xids ~g ~m ~bt ~broadcast:false else [||]
  in
  let shape = List.rev (n :: List.tl (List.rev a_shape)) in
  let outputs = bt * rows * n in
  let (D.Any ddt) = c.xdst in
  let init =
    if not c.xinit then None
    else
      let vs =
        drawn_values st ~name:(D.name ddt) ~low:(-8) ~special:16 outputs
      in
      let made = A.of_array D.Float64 (Iarray.of_list shape) vs in
      let x = A.expect ddt (Contract_cases.cast_to c.xdst (A.Any made)) in
      let held = Contract_cases.cast_to (D.Any D.Float32) (A.Any x) in
      Some (x, A.to_array (A.expect D.Float32 held))
  in
  let dst = A.create (S.device ()) ddt (Iarray.of_list shape) in
  let oa = { Nx_kernel.array = Contract_cases.on bk a; scale = sa } in
  let ob = { Nx_kernel.array = Contract_cases.on bk b; scale = sb } in
  let terms = Option.map (fun d -> Int.max 1 (k + d)) c.xterms in
  let rule = vector_rule ?terms ~rows:segmented ~m:rows ~n ~k oa ob in
  (* The rule but for the terms' bound. *)
  let unbounded =
    Option.is_some (vector_rule ~rows:segmented ~m:rows ~n ~k oa ob)
  in
  (* [xsplit], of a dense key's examples: a launch's budget of 0.6 of the call's
     estimate, so its column groups take several launches. *)
  let split f =
    if not c.xsplit then f ()
    else
      let key =
        match c.xkey with
        | Vf32 -> C.Vector.Dense_f32
        | Vf16 -> C.Vector.Dense_f16
        | Vbf16 -> C.Vector.Dense_bf16
        | Ve2m1 | Vi8 -> fail "a split of a scaled key"
      in
      let e =
        {
          C.batch = bt;
          m = rows;
          n;
          k;
          stage = C.F32;
          nk = true;
          complex = false;
          source = 0;
        }
      in
      match C.Vector.plan e ~key ~rows:segmented ~kn:c.xkn with
      | Some p -> bounded (0.6 *. p.seconds) f
      | None -> fail "no vector plan"
  in
  let bound_terms f =
    match terms with
    | None -> f ()
    | Some t ->
        C.Vector.bound_terms t;
        Fun.protect ~finally:(fun () -> C.Vector.bound_terms max_int) f
  in
  let got, launches =
    split @@ fun () ->
    bound_terms @@ fun () ->
    Plan.traced @@ fun () ->
    let segments =
      if not segmented then None
      else
        let ids_dev =
          Support.on bk (A.of_array D.Int64 (Iarray.of_list [ bt; m ]) ids)
        in
        Some (Nx_kernel.Rows { ids = ids_dev; a_axis; b_axis = 1 })
    in
    Nx_metal.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:
        (Iarray.of_list
           [ (List.length a_shape - 1, if segmented then 3 else 2) ])
      ~segments ~acc:(D.Any D.Float32) ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
      oa ob
  in
  equal answer A.Done got;
  let vector = Option.is_some rule in
  let launched =
    List.filter (fun (t : Plan.traced) -> vector_kernel t.kernel) launches
  in
  equal ~msg:"the vector kernels' launches" bool vector (launched <> []);
  let passes =
    List.exists
      (fun (t : Plan.traced) ->
        match t.kernel with
        | Nx_metal.Kernels.Copy | Copy_8 | Copy_16 | Copy_32 | Copy_64 | Widen
        | Contract_scale ->
            true
        | _ -> false)
      launches
  in
  cover (Printf.sprintf "%s, the vector kernels" (vkey_name c.xkey)) vector;
  cover
    (Printf.sprintf "%s, the tiles" (vkey_name c.xkey))
    ((not vector) && k > 0);
  cover "exchanged, the vector kernels" (rule = Some true);
  cover "a Rows call, the vector kernels" (vector && segmented);
  cover "a row of no group, the vector kernels"
    (vector && Array.exists (fun id -> not (in_group g id)) ids);
  let bound = vkey_bound ~rows:segmented c.xkey in
  let at = if c.xexch then n else rows in
  cover "the key's bound" (vector && at = bound);
  cover "one past the key's bound" ((not vector) && at = bound + 1);
  cover "terms at the lowered terms' bound, the vector kernels"
    (vector && c.xterms = Some 0);
  cover "terms one past the lowered terms' bound, the tiles"
    (unbounded && (not vector) && k > 1 && c.xterms = Some (-1));
  cover "terms one past the lowered terms' bound in a Rows call, the tiles"
    (unbounded && (not vector) && k > 1 && c.xterms = Some (-1) && segmented);
  List.iter
    (fun key ->
      let name = vkey_name key and is = c.xkey = key in
      cover
        (name ^ "'s bound, columns contiguous")
        (is && vector && at = bound && c.xkn);
      cover
        (name ^ "'s bound plus one, columns contiguous, the tiles")
        (is && (not vector) && at = bound + 1 && c.xkn);
      cover
        (name ^ "'s bound of 16 in a Rows call")
        (is && vector && at = 16 && segmented))
    [ Ve2m1; Vi8 ];
  cover "a Rows call of rows of two axes, m first, the vector kernels"
    (vector && c.xpair = Pair_after);
  cover "a Rows call of rows of two axes, m second, the vector kernels"
    (vector && c.xpair = Pair_before);
  cover "255 columns" ((if c.xexch then m else n) = 255);
  cover "a lane of each span" (vector && cdiv k 32 = 1);
  cover "two lanes" (vector && cdiv k 32 = 2);
  cover "16 lanes" (vector && cdiv k 32 = 31);
  cover "32 lanes, some of two spans" (vector && cdiv k 32 > 32);
  cover "a partial span" (vector && k mod 32 <> 0);
  cover "the vector kernels in place" (vector && not passes);
  List.iter
    (fun b ->
      cover
        (Printf.sprintf "scale blocks of %d terms, the vector kernels in place"
           b)
        (vector && c.xblock = b && k mod b = 0 && not passes))
    [ 8; 16; 24; 48 ];
  let forms = vector_forms launched in
  (* kernels.h's NX_CONTRACT_MV_TWINS: a launch of e2m1's, int8 by float16's or
     int8 by float32's kernels at one row a pass takes their twin where its form
     is the twin's (2, 1 and 3), k a multiple of 32 and its lanes 32. *)
  let twin (t : Plan.traced) =
    String.ends_with ~suffix:"_w32" (Nx_metal.Kernels.name t.kernel)
  in
  let lanes32 = cdiv k 32 >= 32 in
  (* Whether [t] launches one of [key]'s n x k kernels at one row a pass, which
     alone have twins. *)
  let of_key (key, _) (t : Plan.traced) =
    List.exists
      (fun other ->
        String.starts_with
          ~prefix:(Printf.sprintf "contract_mv_%s_%s_r1_nk" key other)
          (Nx_metal.Kernels.name t.kernel))
      [ "f32"; "f16"; "bf16" ]
  in
  let keys = [ ("e2m1", 2); ("i8f16", 1); ("i8f32", 3) ] in
  List.iter
    (fun (t : Plan.traced) ->
      let form = word t Nx_metal.Kernels.mv_form in
      let want =
        List.exists (fun ((_, f) as key) -> of_key key t && form = f) keys
        && k mod 32 = 0
        && lanes32
      in
      equal
        ~msg:(Nx_metal.Kernels.name t.kernel ^ ", a twin")
        bool want (twin t))
    launched;
  List.iter
    (fun ((name, _) as key) ->
      let twins = List.exists (fun t -> of_key key t && twin t) launched in
      let general form =
        List.exists
          (fun (t : Plan.traced) ->
            of_key key t
            && (not (twin t))
            && word t Nx_metal.Kernels.mv_form = form)
          launched
      in
      let cover what = cover (Printf.sprintf "%s's %s" name what) in
      cover "twin, n x k" (twins && not c.xexch);
      cover "twin, exchanged" (twins && c.xexch);
      cover "twin, a batch" (twins && bt > 1);
      cover "twin, some lanes of two spans" (twins && cdiv k 32 > 32);
      cover "twin, a Rows call with rows of no group"
        (twins && Array.exists (fun id -> not (in_group g id)) ids);
      cover "twin, an init" (twins && c.xinit);
      cover "twin, a Rows call with rows of no group and an init"
        (twins && c.xinit && Array.exists (fun id -> not (in_group g id)) ids);
      List.iter
        (fun other ->
          let suffix = Printf.sprintf "_%s_r1_nk_typed_w32" other in
          cover
            (Printf.sprintf "twin into a %s dst" other)
            (List.exists
               (fun (t : Plan.traced) ->
                 of_key key t
                 && String.ends_with ~suffix (Nx_metal.Kernels.name t.kernel))
               launched))
        [ "f16"; "bf16" ];
      cover "kernel at 32 lanes, a partial span"
        (List.exists general [ 0; 1; 2; 3 ] && lanes32 && k mod 32 <> 0);
      List.iter
        (fun (form, how) ->
          cover
            (Printf.sprintf "kernel at 32 lanes of whole spans %s" how)
            (general form && lanes32 && k mod 32 = 0))
        (List.filter
           (fun (f, _) -> f <> snd key && (f <> 3 || name <> "e2m1"))
           [
             (2, "by 16-byte words");
             (1, "by 2-byte words");
             (0, "each term where it lies");
             (3, "by 16-byte words, two scale blocks a span");
           ]);
      cover "kernel at 16 lanes, of the twin's form"
        (general (snd key) && cdiv k 32 < 32 && cdiv k 32 >= 16))
    keys;
  cover "whole spans by 16-byte words" (List.mem 2 forms);
  cover "whole spans by 2-byte words" (List.mem 1 forms);
  cover "whole spans by 16-byte words, two scale blocks a span"
    (List.mem 3 forms);
  cover "each term where it lies, n x k" (List.mem 0 forms);
  cover "float4_e2m1fn's whole spans by 16-byte words"
    (c.xkey = Ve2m1 && List.mem 2 forms);
  let halves, both = if c.xexch then e2m1_reach c a sa else e2m1_reach c b sb in
  cover "float4_e2m1fn's ±0.5, the vector kernels" (vector && halves);
  cover "codes within and outside [2, 240] in one call, the vector kernels"
    (vector && both);
  cover "int8's whole spans by 2-byte words" (c.xkey = Vi8 && List.mem 1 forms);
  cover "several vector launches" (List.length launched > 1);
  cover "several vector launches of a Rows call with an init"
    (List.length launched > 1 && segmented && c.xinit);
  cover "16 lanes over columns contiguous"
    (vector && c.xkn && (not c.xexch) && cdiv k 32 >= 16 && cdiv k 32 < 32);
  cover "a dense weight of columns contiguous, the vector kernels"
    (vector && c.xkn && (not c.xexch) && List.mem c.xkey [ Vf32; Vf16; Vbf16 ]);
  (* A scaled weight of columns contiguous: its k x n kernels, a lane of 8
     columns, 4 rows a pass but for one row and Rows calls, its scale blocks
     found by a counter, its lane's row read by mv_load's 4-, 2- or 1-byte
     loads. *)
  let scaled_kn =
    vector && c.xkn && (not c.xexch) && List.mem c.xkey [ Ve2m1; Vi8 ]
  in
  let in_place = scaled_kn && not passes in
  let general, mixed, last =
    if in_place then kn_codes c ~n sb else (false, false, false)
  in
  cover "an e2m1 weight of columns contiguous, a run of a code outside [2, 240]"
    general;
  cover "an e2m1 weight of columns contiguous, a run of codes on both sides"
    mixed;
  cover "an e2m1 weight of columns contiguous, a lane past n of a code outside"
    last;
  (* Exchanged, b k x n too is copied as the other operand; the weight takes no
     products pass. *)
  let products =
    List.exists
      (fun (t : Plan.traced) -> t.kernel = Nx_metal.Kernels.Contract_scale)
      launches
  in
  cover
    "a scaled weight of columns contiguous exchanged, the vector kernels in \
     place"
    (vector && c.xkn && c.xexch
    && List.mem c.xkey [ Ve2m1; Vi8 ]
    && not products);
  cover "an e2m1 weight of columns contiguous and odd rows, its products"
    (vector && c.xkn && (not c.xexch) && c.xkey = Ve2m1 && passes && n mod 2 = 1);
  let wide = if c.xexch then a else b in
  let (A.Any wx) = wide in
  let wl = A.layout wx in
  let wbits = D.bits (A.dtype wx) in
  let terms = List.length (if c.xexch then [ bt; m; k ] else b_shape) - 1 in
  let at = A.Layout.offset wl * wbits / 8
  and row = A.Layout.stride wl terms * wbits / 8 in
  let align =
    List.fold_left
      (fun g x -> if x = 0 then g else Int.min g (x land -x))
      16 [ at; row ]
  in
  List.iter
    (fun key ->
      cover
        (Printf.sprintf "%s of columns contiguous, the vector kernels in place"
           (vkey_name key))
        (in_place && c.xkey = key))
    [ Ve2m1; Vi8 ];
  cover "a scaled weight of columns contiguous, a partial pass of 4 rows"
    (in_place && (not segmented) && rows > 4 && rows mod 4 <> 0);
  cover "a scaled weight of columns contiguous, a lane past n"
    (in_place && n mod 8 <> 0);
  cover "a scaled weight of columns contiguous, scale blocks across spans"
    (in_place && (c.xblock = 24 || c.xblock = 48) && k > c.xblock);
  cover "a scaled weight of columns contiguous, a scale a term"
    (in_place && c.xblock = 1);
  cover "a scaled weight of columns contiguous, one scale block"
    (in_place && c.xblock = 0);
  cover "a scaled weight of columns contiguous, scale blocks of columns"
    (in_place && c.xrn > 1);
  cover "a scaled weight of columns contiguous, a Rows call"
    (in_place && segmented);
  List.iter
    (fun (bytes, what) ->
      cover
        (Printf.sprintf "a scaled weight of columns contiguous, rows at %s" what)
        (in_place && align = bytes))
    [ (1, "an odd byte"); (2, "2 bytes") ];
  cover "a scaled weight of columns contiguous, rows at 4 bytes"
    (in_place && align >= 4);
  cover "a Rows call over columns contiguous, the vector kernels"
    (vector && c.xkn && segmented);
  cover "the vector kernels after a copy or products" (vector && passes);
  (* nx_metal.mli: a vector call's sums take a last pass into dst where there is
     an init or dst is of neither float32 nor its other operand's unscaled
     float16 or bfloat16; here dst shares no memory with an operand. *)
  let last =
    List.exists
      (fun (t : Plan.traced) -> t.kernel = Nx_metal.Kernels.Contract_finish)
      launches
  in
  let typed =
    c.xdst = D.Any D.Float32 || ((not c.xscaled) && c.xdst = c.xvec)
  in
  if vector then
    equal ~msg:"a vector call's last pass" bool (c.xinit || not typed) last;
  cover "the vector kernels into dst" (vector && not last);
  List.iter
    (fun (D.Any d as dt) ->
      cover
        (Printf.sprintf "the vector kernels into a %s dst" (D.name d))
        (vector && (not last) && c.xdst = dt);
      cover
        (Printf.sprintf "a %s dst of another dtype, the vector kernels"
           (D.name d))
        (vector && (not c.xinit) && c.xdst = dt && last))
    [ D.Any D.Float16; D.Any D.Bfloat16 ];
  cover "a scaled other of dst's dtype, a last pass"
    (vector && c.xscaled && (not c.xinit) && c.xdst = c.xvec
   && c.xdst <> D.Any D.Float32 && last);
  cover "an init, the vector kernels" (vector && c.xinit);
  (* The tiles' plan, at the stage the operands give, for the other calls. *)
  let ranges =
    if vector then 0
    else
      let stage =
        if Option.is_some sa || Option.is_some sb then C.F32
        else
          let (A.Any x) = a in
          let (A.Any y) = b in
          converted_stage (D.Any (A.dtype x)) (D.Any (A.dtype y))
      in
      let e =
        {
          C.batch = bt;
          m = rows;
          n;
          k;
          stage;
          nk = true;
          complex = false;
          source = 0;
        }
      in
      match if segmented then C.grouped e ~groups:g else C.plan e with
      | Some p -> p.ranges
      | None -> fail "no tiles' plan"
  in
  (* The operands' values, each read once: a term's index is its row's, or its
     column's, number in its shape with kk 0, plus kk. *)
  let af = tabulated a_shape av and bf = tabulated b_shape bv in
  let sums =
    Array.init outputs (fun o ->
        let z = o / (rows * n) and r = o / n mod rows and j = o mod n in
        let i, sub =
          if c.xpair = Pair_before then (r mod m, r / m) else (r / q, r mod q)
        in
        let init = Option.map (fun (_, v) -> v.(o)) init in
        let b0 =
          if not segmented then number_of b_shape [ z; j; 0 ]
          else number_of b_shape [ z; Int64.to_int ids.((z * m) + i); j; 0 ]
        in
        let a0 =
          match c.xpair with
          | No_pair -> number_of a_shape [ z; i; 0 ]
          | Pair_after -> number_of a_shape [ z; i; sub; 0 ]
          | Pair_before -> number_of a_shape [ z; sub; i; 0 ]
        in
        if segmented && not (in_group g ids.((z * m) + i)) then
          Option.value init ~default:0.
        else if vector then vectored_at ~k ~init af a0 bf b0
        else associated_at ~ranges ~k ~init af a0 1 bf b0 1)
  in
  let as64 x =
    A.to_array (A.expect D.Float64 (Contract_cases.cast_to (D.Any D.Float64) x))
  in
  let held = A.Any (A.of_array D.Float32 (Iarray.of_list shape) sums) in
  let want = as64 (Contract_cases.cast_to c.xdst held) in
  equal (array float_exact) want (as64 (A.Any (Support.host dst)))

let vcall_of ?(xblock = 32) ?(xrn = 1) ?(xsd = D.Any D.Float16)
    ?(xvec = D.Any D.Float32) ?(xscaled = false) ?(xexch = false) ?xgroups
    ?(xids = Each) ?(xkn = false) ?(xoff = false) ?(xwskip = 0) ?(xinit = false)
    ?(xdst = D.Any D.Float32) ?(xsplit = false) ?(xpair = No_pair) ?xterms
    ?(xseed = 3) xkey xbt xm xn xk =
  {
    xkey;
    xblock;
    xrn;
    xsd;
    xvec;
    xscaled;
    xexch;
    xgroups;
    xids;
    xbt;
    xm;
    xn;
    xk;
    xkn;
    xoff;
    xwskip;
    xinit;
    xdst;
    xsplit;
    xpair;
    xterms;
    xseed;
  }

(* Each key at its bound and one past, columns at 256 and 255, exchanged
   too; lanes of 1, 2, 16 and 32; a partial span; a Rows call with rows
   of no group, and with rows of two axes, m first or second; e2m1's,
   int8 by float16's and int8 by float32's (at scale blocks of 16
   terms) twins n x k, exchanged, of a batch, of lanes of two spans, of
   a Rows call with rows of no group, with an init and typed, and their
   kernels at a partial span (k 1008 with a scale a
   column, so that the weight is in place; an odd k's e2m1, 1023, is a
   copy's), at the other forms and at 16 lanes, int8 started 2 bytes in
   for its 2-byte words; the operands in place, copied, at an odd offset and with
   their columns contiguous, at 1, 2, 4, 8, 16 and 32 lanes and in a
   Rows call; scale blocks of 8, 16, 24 and 48 terms, two of them a
   span at 16 and 24; launches
   cut by a smaller budget, a Rows call's with an init among them, whose
   rows' buckets the launches write; scales of blocks of one term, which a
   products pass takes; an init and dsts of each float dtype; dsts of
   the other operand's float16 and bfloat16, which the vector kernels
   write, n x k, k x n, exchanged and of a Rows call with rows of no
   group, and of another dtype or a scaled other's, which a last pass
   writes. *)
(* contract.mli: every call the vector kernels take finds launches at
   the library's budget: each key at its bound of terms, at every row
   count to its bound, n x k and k x n, of a Rows call or not, plans
   launches. *)
let test_vector_terms (key, rows, kn, m) =
  let k = C.Vector.terms key ~rows in
  let e =
    {
      C.batch = 1;
      m;
      n = C.Vector.columns;
      k;
      stage = C.F32;
      nk = true;
      complex = false;
      source = 0;
    }
  in
  let found =
    match C.Vector.plan e ~key ~rows ~kn with
    | Some v when Array.length v.launches > 0 -> "launches"
    | Some _ | None -> "no launch"
  in
  equal string "launches" found

(* nx_metal.mli: a call of one term past its key's bound takes the tiles, which
   find launches: each key at its bound of rows, n x k and k x n, of a Rows call
   or not, plans as the tiles plan it, a scaled key's as the loaders read it,
   from float32 rows. Plans alone: the weight would be tens of megabytes. *)
let test_tiles_past_terms (key, rows, kn) =
  let m = C.Vector.bound key ~rows and k = C.Vector.terms key ~rows + 1 in
  let stage, source =
    match key with
    | C.Vector.Dense_f32 -> (C.F32, 0)
    | Dense_f16 -> (C.F16, 0)
    | Dense_bf16 -> (C.Bf16, 0)
    | E2m1_by_exponent | Int8_by_value -> (C.F32, 4)
  in
  let e =
    { C.batch = 1; m; n = 8192; k; stage; nk = not kn; complex = false; source }
  in
  let plan = if rows then C.grouped e ~groups:1 else C.plan e in
  equal ~msg:"a plan" bool true (Option.is_some plan)

let vector_terms_cases =
  List.concat_map
    (fun key ->
      List.concat_map
        (fun rows -> [ (key, rows, false); (key, rows, true) ])
        [ false; true ])
    C.Vector.
      [ Dense_f32; Dense_f16; Dense_bf16; E2m1_by_exponent; Int8_by_value ]

let vector_terms_id (key, rows, kn) =
  Printf.sprintf "%s%s%s"
    (match key with
    | C.Vector.Dense_f32 -> "float32"
    | Dense_f16 -> "float16"
    | Dense_bf16 -> "bfloat16"
    | E2m1_by_exponent -> "float4_e2m1fn"
    | Int8_by_value -> "int8")
    (if rows then ", a Rows call" else "")
    (if kn then ", k x n" else "")

let vector_rows_cases =
  List.concat_map
    (fun (key, rows, kn) ->
      List.init (C.Vector.bound key ~rows) (fun i -> (key, rows, kn, i + 1)))
    vector_terms_cases

let vector_rows_id (key, rows, kn, m) =
  Printf.sprintf "%s, %d rows" (vector_terms_id (key, rows, kn)) m

(* What a call of [k] float16 terms by [n] float16 columns plans: a's terms and
   b's of stride 0 read one element each, so the operands take 2 and [2 n]
   bytes. Planned only. *)
let terms_plan (k, n) =
  let dv = S.device () in
  let over dt n shape strides =
    let x = A.create dv dt (Iarray.of_list [ n ]) in
    A.v dt
      (L.v ~strides:(Iarray.of_list strides) (Iarray.of_list shape))
      (A.buffer x)
  in
  let a = over D.Float16 1 [ 1; k ] [ 0; 0 ] in
  let b = over D.Float16 n [ k; n ] [ 0; 1 ] in
  let dst = A.create dv D.Float32 (Iarray.of_list [ 1; n ]) in
  let operand x = { Nx_kernel.array = A.Any x; scale = None } in
  match
    C.call
      ~ops:[| A.Any dst; A.Any a; A.Any b |]
      ~batch:(Iarray.of_list [])
      ~contracting:(Iarray.of_list [ (1, 0) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst:(A.Any dst) ~init:None
      (operand a) (operand b)
  with
  | Error _ -> "declined"
  | Ok (C.Recast _) -> "in pieces"
  | Ok (C.Planned None) -> "no output"
  | Ok (C.Planned (Some p)) ->
      let vector (l : Plan.launch) =
        String.starts_with ~prefix:"contract_mv_"
          (Nx_metal.Kernels.name l.kernel)
      in
      if Array.exists vector p.launches then "the vector kernels"
      else "the tiles"

(* nx_metal.mli: a float16 call the vector kernels take, of 256 columns, takes
   the tiles' path past 2^22 terms; a call of one column takes it at any
   count. *)
let terms_cases =
  [
    ("2^22 terms by 256 columns", (1 lsl 22, 256), "the vector kernels");
    ("2^22 + 1 terms by 256 columns", ((1 lsl 22) + 1, 256), "the tiles");
    ("2^24 terms by 256 columns", (1 lsl 24, 256), "the tiles");
    ("2^32 terms by 256 columns", (1 lsl 32, 256), "the tiles");
    ("2^32 terms by a column", (1 lsl 32, 1), "the tiles");
    ("2^33 terms by a column", (1 lsl 33, 1), "the tiles");
  ]

let terms_id (name, _, _) = name
let test_terms (_, k, want) = equal string want (terms_plan k)

let vcall_examples =
  let q16 = vcall_of ~xblock:16 ~xsd:(D.Any D.Float32) in
  List.concat_map
    (fun key ->
      let b = vkey_bound ~rows:false key in
      [
        vcall_of ~xterms:0 key 1 1 256 96;
        vcall_of ~xterms:(-1) key 1 1 256 96;
        vcall_of ~xterms:(-1) ~xkn:true key 1 1 256 96;
        vcall_of ~xterms:(-1) ~xgroups:2 key 1 1 256 96;
        vcall_of key 1 b 256 64;
        vcall_of key 1 (b + 1) 256 64;
        vcall_of key 1 b 255 64;
        vcall_of ~xexch:true key 1 300 b 64;
        vcall_of ~xexch:true key 1 300 (b + 1) 64;
      ])
    [ Vf32; Vf16; Vbf16; Ve2m1; Vi8 ]
  @ List.concat_map
      (fun key ->
        let b = vkey_bound ~rows:false key in
        [
          vcall_of ~xkn:true key 1 b 256 64;
          vcall_of ~xkn:true key 1 (b + 1) 256 64;
          vcall_of ~xgroups:2 key 1 16 256 64;
        ])
      [ Ve2m1; Vi8 ]
  @ [
      vcall_of Vf16 1 1 300 32;
      vcall_of Vf16 1 2 257 33;
      vcall_of Ve2m1 1 3 256 992;
      vcall_of Vi8 2 3 300 1056;
      vcall_of ~xvec:(D.Any D.Bfloat16) Ve2m1 1 1 520 2880;
      vcall_of ~xsd:(D.Any D.Float32) ~xvec:(D.Any D.Float16) Vi8 1 3 300 100;
      vcall_of ~xblock:0 Vi8 1 3 300 100;
      vcall_of ~xgroups:3 ~xids:Outside ~xinit:true Vi8 1 8 256 64;
      vcall_of ~xgroups:2 ~xids:Outside Ve2m1 2 5 300 96;
      vcall_of ~xpair:Pair_after ~xgroups:2 ~xids:Outside Ve2m1 2 3 300 96;
      vcall_of Ve2m1 1 4 256 1024;
      vcall_of ~xexch:true Ve2m1 1 300 4 1024;
      vcall_of Ve2m1 2 3 256 1056;
      vcall_of ~xgroups:3 ~xids:Outside Ve2m1 1 4 256 1024;
      vcall_of ~xvec:(D.Any D.Float16) ~xdst:(D.Any D.Float16) Ve2m1 1 3 256
        1024;
      vcall_of ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16) Ve2m1 1 3 256
        1024;
      vcall_of Ve2m1 1 3 256 1023;
      vcall_of ~xblock:0 Ve2m1 1 3 256 1008;
      vcall_of ~xwskip:4 Ve2m1 1 3 256 1024;
      vcall_of ~xwskip:2 Ve2m1 1 3 256 1024;
      vcall_of ~xgroups:3 ~xids:Outside ~xinit:true Ve2m1 1 4 256 1024;
      vcall_of ~xinit:true Ve2m1 1 4 256 1024;
      vcall_of ~xwskip:2 Vi8 1 1 256 1024;
      vcall_of ~xwskip:2 ~xexch:true Vi8 1 300 1 1024;
      vcall_of ~xwskip:2 Vi8 2 1 256 1056;
      vcall_of ~xwskip:2 ~xgroups:3 ~xids:Outside Vi8 1 4 256 1024;
      vcall_of ~xwskip:2 ~xgroups:3 ~xids:Outside ~xinit:true Vi8 1 4 256 1024;
      vcall_of ~xwskip:2 ~xinit:true Vi8 1 1 256 1024;
      vcall_of ~xwskip:2 ~xvec:(D.Any D.Float16) ~xdst:(D.Any D.Float16) Vi8 1 1
        256 1024;
      vcall_of ~xwskip:2 ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16) Vi8 1
        1 256 1024;
      vcall_of ~xblock:0 ~xwskip:2 Vi8 1 1 256 1008;
      vcall_of Vi8 1 1 256 1024;
      vcall_of ~xwskip:1 Vi8 1 1 256 1024;
      vcall_of ~xwskip:2 Vi8 1 1 256 992;
      q16 Vi8 1 1 256 1024;
      q16 ~xexch:true Vi8 1 300 1 1024;
      q16 Vi8 2 1 256 1056;
      q16 ~xgroups:3 ~xids:Outside Vi8 1 4 256 1024;
      q16 ~xgroups:3 ~xids:Outside ~xinit:true Vi8 1 4 256 1024;
      q16 ~xinit:true Vi8 1 1 256 1024;
      q16 ~xvec:(D.Any D.Float16) ~xdst:(D.Any D.Float16) Vi8 1 1 256 1024;
      q16 ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16) Vi8 1 1 256 1024;
      q16 Vi8 1 1 256 1008;
      q16 Vi8 1 1 256 992;
      q16 ~xwskip:2 Vi8 1 1 256 1024;
      vcall_of ~xsd:(D.Any D.Float32) Vi8 1 1 256 1024;
      vcall_of ~xsd:(D.Any D.Float32) ~xwskip:2 Vi8 1 1 256 1024;
      vcall_of ~xsd:(D.Any D.Float32) ~xwskip:1 Vi8 1 1 256 1024;
      vcall_of ~xblock:0 ~xsd:(D.Any D.Float32) Vi8 1 1 256 1008;
      vcall_of ~xblock:16 Vi8 1 1 256 1024;
      vcall_of ~xpair:Pair_before ~xgroups:2 ~xids:Outside Ve2m1 2 3 300 96;
      vcall_of ~xpair:Pair_before ~xgroups:3 ~xids:Outside ~xinit:true Vi8 1 5
        256 64;
      vcall_of ~xkn:true ~xpair:Pair_before ~xgroups:3 ~xids:Outside Vf16 1 1
        300 96;
      (* Scaled weights of columns contiguous: MXFP4's 11 rows, passes of 4 and
         a partial one; a pass of 1 past them, a lane of 4 columns, scale blocks
         of 48 across spans, of one term and one block, of 2 columns; a lane's
         rows at an odd byte and at 2 bytes; a Rows call with an init; typed
         dsts; one row. *)
      vcall_of ~xkn:true ~xvec:(D.Any D.Bfloat16) Ve2m1 1 11 520 2880;
      vcall_of ~xkn:true ~xblock:48 Ve2m1 1 5 300 100;
      vcall_of ~xkn:true ~xblock:48 ~xrn:2 ~xsd:(D.Any D.Float32) Vi8 1 3 257
        1056;
      vcall_of ~xkn:true ~xblock:1 Ve2m1 1 4 256 64;
      vcall_of ~xkn:true ~xblock:0 Vi8 1 3 256 100;
      vcall_of ~xkn:true ~xrn:2 Ve2m1 1 2 300 96;
      vcall_of ~xkn:true ~xwskip:2 Ve2m1 1 4 256 64;
      vcall_of ~xkn:true ~xwskip:1 Vi8 1 3 256 64;
      vcall_of ~xkn:true ~xwskip:2 Vi8 1 3 256 64;
      vcall_of ~xkn:true ~xgroups:3 ~xids:Outside ~xinit:true Ve2m1 1 4 256 64;
      vcall_of ~xkn:true ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16) Ve2m1
        1 11 256 64;
      vcall_of ~xkn:true ~xvec:(D.Any D.Float16) ~xdst:(D.Any D.Float16) Vi8 1 1
        256 64;
      vcall_of ~xkn:true ~xexch:true Ve2m1 1 300 4 64;
      vcall_of ~xkn:true Ve2m1 1 4 257 64;
      vcall_of ~xgroups:4 ~xids:One Vf16 1 2 256 64;
      vcall_of ~xoff:true Vf16 1 2 300 96;
      vcall_of ~xoff:true Ve2m1 1 4 256 64;
      vcall_of ~xkn:true Vbf16 1 2 300 96;
      vcall_of ~xkn:true Vf32 1 1 300 1;
      vcall_of ~xkn:true Vf16 1 2 257 64;
      vcall_of ~xkn:true Vf16 1 1 520 160;
      vcall_of ~xkn:true Vbf16 1 2 300 300;
      vcall_of ~xkn:true Vf32 2 1 256 1056;
      vcall_of ~xkn:true ~xoff:true Vf16 1 2 300 64;
      vcall_of ~xkn:true Vf16 1 2 300 992;
      vcall_of ~xblock:8 Vi8 1 3 256 64;
      vcall_of ~xblock:16 Ve2m1 1 2 300 96;
      vcall_of ~xblock:24 Vi8 1 3 300 96;
      vcall_of ~xblock:48 Ve2m1 1 1 256 1056;
      vcall_of ~xblock:24 ~xgroups:2 Vi8 1 3 300 96;
      vcall_of Vi8 1 3 300 64;
      vcall_of ~xexch:true Vi8 1 300 3 64;
      vcall_of ~xsplit:true Vf32 2 1 520 2880;
      vcall_of ~xsplit:true Vf16 1 2 520 2880;
      vcall_of ~xsplit:true ~xgroups:2 ~xids:Outside ~xinit:true Vf16 1 2 520
        2880;
      vcall_of ~xkn:true ~xgroups:3 ~xids:Outside ~xinit:true Vf16 1 2 300 96;
      vcall_of ~xkn:true Vi8 1 3 256 64;
      vcall_of ~xblock:1 Ve2m1 1 4 256 64;
      vcall_of ~xblock:0 ~xsd:(D.Any D.Float32) Vi8 1 3 256 40;
      vcall_of ~xscaled:true Vf32 1 1 256 64;
      vcall_of ~xinit:true ~xdst:(D.Any D.Float16) Vf32 2 1 300 100;
      vcall_of ~xinit:true ~xdst:(D.Any D.Bfloat16) ~xexch:true Vi8 1 256 3 64;
      vcall_of ~xvec:(D.Any D.Float16) ~xdst:(D.Any D.Float16) Ve2m1 1 3 256 992;
      vcall_of ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16) ~xgroups:2
        ~xids:Outside Vi8 1 5 300 96;
      vcall_of ~xvec:(D.Any D.Float16) ~xdst:(D.Any D.Float16) ~xkn:true Vf16 1
        2 300 96;
      vcall_of ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16) ~xexch:true
        Vf32 1 300 4 64;
      vcall_of ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16) Ve2m1 1 1 520
        2880;
      vcall_of ~xdst:(D.Any D.Float16) Ve2m1 1 3 256 992;
      vcall_of ~xvec:(D.Any D.Float16) ~xdst:(D.Any D.Bfloat16) Vi8 1 3 300 100;
      vcall_of ~xscaled:true ~xvec:(D.Any D.Bfloat16) ~xdst:(D.Any D.Bfloat16)
        Vf32 1 1 256 64;
    ]

(* nx_metal.mli: a weight as a model stores it, its terms contiguous, at rows to
   its key's bound, 11 for MXFP4's e2m1 and 3 for int8, and 256 columns or more,
   takes the vector kernels, which read it and its scales where they lie: no
   copy, widening or scale pass. Each output is the sum of a's elements times
   the weight's values, exact here in float32, rounded once into dst. *)
let vector_weights =
  [ (Mxfp4, 1); (Mxfp4, 11); (Q8_0, 1); (Q8_0, 3); (Q6_k, 3) ]

(* kernels.h's NX_MV_FORM the host gives each weight, a's rows starting on 16
   bytes: MXFP4's rows of codes start on 16 bytes and its scale blocks are
   spans, so 16-byte words; Q8_0's quants start 2 bytes into 34-byte blocks, so
   2-byte words; Q6_K's quants start on 16 bytes and its scale blocks of 16
   terms split a span, so 16-byte words with two scale blocks a span. *)
let vector_form_of = function Mxfp4 -> 2 | Q8_0 -> 1 | Q6_k -> 3

let test_vector_weight (w, m) =
  let bk = backend () in
  let n = 300 and k = 128 in
  let b, scale, inner, value, adt = weight_of bk ~n ~k w in
  let a_at r t = Float.of_int (((r + t) mod 7) - 3) in
  let a_values = Array.init (m * k) (fun i -> a_at (i / k) (i mod k)) in
  let a =
    Contract_cases.on bk
      (Contract_cases.cast_to adt
         (A.Any (A.of_array D.Float64 (Iarray.of_list (m :: inner)) a_values)))
  in
  let p = List.length inner in
  let contracting = Iarray.of_list (List.init p (fun i -> (1 + i, 1 + i))) in
  let (D.Any ddt) = adt in
  let dst = A.create (S.device ()) ddt (Iarray.of_list [ m; n ]) in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract ~batch:(Iarray.of_list []) ~contracting ~segments:None
      ~acc:(D.Any D.Float32) ~dst ~init:None
      { Nx_kernel.array = a; scale = None }
      { Nx_kernel.array = b; scale = Some scale }
  in
  equal answer A.Done got;
  let names p =
    List.filter_map
      (fun (t : Plan.traced) ->
        if p t.kernel then Some (Nx_metal.Kernels.name t.kernel) else None)
      launches
  in
  equal ~msg:"the vector kernels' launches" int 1
    (List.length (names vector_kernel));
  equal ~msg:"copies, widenings and scale passes" (list string) []
    (names (fun k -> apart_copy k || k = Widen || k = Contract_scale));
  equal ~msg:"the vector launch's load form" (list int)
    [ vector_form_of w ]
    (vector_forms launches);
  let sums =
    Array.init (m * n) (fun o ->
        let r = o / n and j = o mod n in
        let s = ref 0. in
        for t = 0 to k - 1 do
          s := !s +. (a_at r t *. value j t)
        done;
        !s)
  in
  let as64 x =
    A.to_array (A.expect D.Float64 (Contract_cases.cast_to (D.Any D.Float64) x))
  in
  let want =
    as64
      (Contract_cases.cast_to adt
         (A.Any (A.of_array D.Float64 (Iarray.of_list [ m; n ]) sums)))
  in
  equal (array float_exact) want (as64 (A.Any (Support.host dst)))

(* nx_metal.mli: a dense weight of columns contiguous, as [x @ W] with [W] of
   [[k; n]], at a key's bound of rows, takes the vector kernels where it lies:
   no copy. Each output is the exact sum. *)
let vector_dense =
  D.[ (Any Float32, 1); (Any Float16, 1); (Any Float16, 2); (Any Bfloat16, 2) ]

let dense_id (D.Any d, m) = Printf.sprintf "%s, %d rows" (D.name d) m

let test_vector_dense (wdt, m) =
  let bk = backend () in
  let n = 300 and k = 96 in
  let a_at r t = Float.of_int (((r + t) mod 7) - 3) in
  let w_at t j = Float.of_int ((((t * 3) + j) mod 5) - 2) in
  let host dt shape f =
    Contract_cases.on bk
      (Contract_cases.cast_to dt
         (A.Any (A.of_array D.Float64 (Iarray.of_list shape) f)))
  in
  let a =
    host wdt [ m; k ] (Array.init (m * k) (fun i -> a_at (i / k) (i mod k)))
  in
  let w =
    host wdt [ k; n ] (Array.init (k * n) (fun i -> w_at (i / n) (i mod n)))
  in
  let dst = A.create (S.device ()) D.Float32 (Iarray.of_list [ m; n ]) in
  let got, launches =
    Plan.traced @@ fun () ->
    Nx_metal.contract ~batch:(Iarray.of_list [])
      ~contracting:(Iarray.of_list [ (1, 0) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst ~init:None
      { Nx_kernel.array = a; scale = None }
      { Nx_kernel.array = w; scale = None }
  in
  equal answer A.Done got;
  let names p =
    List.filter_map
      (fun (t : Plan.traced) ->
        if p t.kernel then Some (Nx_metal.Kernels.name t.kernel) else None)
      launches
  in
  equal ~msg:"the vector kernels' launches" int 1
    (List.length (names vector_kernel));
  equal ~msg:"copies and finishes" (list string) []
    (names (fun k -> apart_copy k || k = Contract_finish));
  let want =
    Array.init (m * n) (fun o ->
        let r = o / n and j = o mod n in
        let s = ref 0. in
        for t = 0 to k - 1 do
          s := !s +. (a_at r t *. w_at t j)
        done;
        !s)
  in
  equal (array float_exact) want (A.to_array (Support.host dst))

(* Rows of operands of other dtypes *)

type rconverted = {
  rv : rbits;
  rva : D.any;
  rvb : D.any;
  rbview : cvview;
  rsa : sspec option;
  rsb : sspec option;
}

let pp_rconverted ppf c =
  let name (D.Any d) = D.name d in
  Format.fprintf ppf "%a, %s x %s into float32%s%s" pp_rbits c.rv (name c.rva)
    (name c.rvb) (cvview_name c.rbview)
    (String.concat ""
       (List.map
          (fun (what, s) ->
            Option.fold ~none:""
              ~some:(Format.asprintf ", %s scaled by %a" what pp_sspec)
              s)
          [ ("a", c.rsa); ("b", c.rsb) ]))

let rconverted_case =
  Gen.(
    with_pp pp_rconverted
      (let pick = of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) held_dtypes in
       let+ rv = rbits_case
       and+ rva = pick
       and+ rvb = pick
       and+ rbview = cvview_case true in
       let rvb =
         if rva = rvb && List.mem rva native then D.Any D.Int8 else rvb
       in
       let rstage = if rv.rstage = 3 then 0 else rv.rstage in
       {
         rv = { rv with rint = false; rstage };
         rva;
         rvb;
         rbview;
         rsa = None;
         rsb = None;
       }))

(* Rows calls of a scaled a, b or both, whose batch axis is not split, since a
   scale would split with it. *)
let rscaled_case =
  Gen.(
    with_pp pp_rconverted
      (let* c = rconverted_case in
       let rank_a = if c.rv.rsplit then 4 else 3 in
       let+ which = int_range 1 3
       and+ sa = sspec_case rank_a
       and+ sb = sspec_case 4 in
       let rview = if c.rv.rview = Split_ids then Fresh_ids else c.rv.rview in
       {
         c with
         rv = { c.rv with rview };
         rsa = (if which land 1 <> 0 then Some sa else None);
         rsb = (if which land 2 <> 0 then Some sb else None);
       }))

(* nx_kernel.mli and nx_metal.mli: a Rows call into float32 of operands of any
   dtypes but float64 and complex128 multiplies each row by its group's b, as
   the call of its operands' casts: every output of a row in a group has the
   stated association of the cast operands' products, the plan's of the stage
   its dtypes give, then init; one of a row of no group is init alone, or +0. *)
let law_rows_converted w =
  let c = w.rv in
  let bk = backend () in
  let stage =
    if Option.is_some w.rsa || Option.is_some w.rsb then C.F32
    else converted_stage w.rva w.rvb
  in
  let st = Random.State.make [| c.rseed |] in
  let sa, sb, sy = rows_shapes c in
  let count = List.fold_left ( * ) 1 in
  let a_view, b_view =
    match w.rbview with
    | Stepped_groups -> (`Fresh, `Step 1)
    | v -> views_of v ~a_m:1 ~b_major:2
  in
  let special = Int.max 64 (4 * c.rk) in
  let a_any, av = laid_operand ~special st bk w.rva sa ~low:(-8) a_view in
  let b_any, bv = laid_operand ~special st bk w.rvb sb ~low:(-8) b_view in
  let (A.Any a) = a_any in
  let (A.Any b) = b_any in
  let scale_a = Option.map (fun sp -> drawn_scale st bk sp sa) w.rsa in
  let scale_b = Option.map (fun sp -> drawn_scale st bk sp sb) w.rsb in
  let scaled_by shape scale vs =
    match scale with
    | None -> vs
    | Some (_, f) -> Array.mapi (fun t v -> f (index_of shape t) v) vs
  in
  let av = scaled_by sa scale_a av and bv = scaled_by sb scale_b bv in
  let ids = rows_ids st c in
  let outputs = count sy in
  let q = if c.rsplit then 3 else 1 in
  let rows = q * c.rm and n = c.rn and k = c.rk in
  let id_of o = ids.((o / (rows * n) * c.rm) + (o / n mod rows / q)) in
  let init =
    if c.rinit then
      Some
        (stored D.Float32 sy
           (drawn_values st ~name:"float32" ~low:(-8) ~special:16 outputs))
    else None
  in
  let dst =
    Support.on bk (fst (stored D.Float32 sy (Array.make outputs nan)))
  in
  let p = rows_plan ~stage c in
  let products = k > 0 && c.rg > 0 in
  let scaled = Option.is_some w.rsa || Option.is_some w.rsb in
  if not scaled then begin
    cover "float16" (stage = C.F16);
    cover "bfloat16" (stage = C.Bf16);
    cover "float32" (stage = C.F32);
    cover "32x64" (products && p.tile = C.G32);
    cover "16x64" (products && p.tile = C.G16);
    let cast (D.Any d) = D.Any d <> stage_dtype ~scaled:false w.rva w.rvb in
    let recast = cast w.rva || cast w.rvb in
    cover "32x64, a cast" (products && recast && p.tile = C.G32);
    cover "16x64, a cast" (products && recast && p.tile = C.G16);
    cover_codes [ (w.rva, av); (w.rvb, bv) ]
  end;
  cover "an operand narrower than a byte"
    (List.exists
       (fun d -> List.mem d [ w.rva; w.rvb ])
       D.[ Any Int4; Any Uint4; Any Bit; Any Float4_e2m1fn ]);
  cover "b n x k" c.rnk;
  cover "ids outside [0, g)" (c.rids = Outside);
  cover "several ranges" (products && p.ranges > 1);
  cover "an init" c.rinit;
  if scaled then begin
    let specs = List.filter_map Fun.id [ w.rsa; w.rsb ] in
    cover "Exponent" (List.exists (fun s -> s.sexp) specs);
    cover "Value" (List.exists (fun s -> not s.sexp) specs);
    cover "a scaled" (Option.is_some w.rsa && Option.is_none w.rsb);
    cover "b scaled" (Option.is_none w.rsa && Option.is_some w.rsb);
    cover "both scaled" (Option.is_some w.rsa && Option.is_some w.rsb);
    let shared s = List.nth s.sratios 1 = 1 && c.rg > 1 in
    cover "a scale shared by the groups"
      (Option.fold ~none:false ~some:shared w.rsb);
    let scales = List.filter_map (Option.map fst) [ scale_a; scale_b ] in
    cover "a scale at an offset"
      (List.exists (fun s -> past_start (scale_layout s)) scales);
    cover "a stepped scale"
      (List.exists (fun s -> skips (scale_layout s)) scales)
  end;
  let got, launches =
    rows_call ?sa:(Option.map fst scale_a) ?sb:(Option.map fst scale_b) c ~a ~b
      ~ids ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
  in
  equal answer A.Done got;
  let ran prefix =
    List.exists (fun (t : Plan.traced) -> tile_prefix prefix t.kernel) launches
  in
  let keyed =
    Option.is_none w.rsa
    && loads ~a:(entered w.rva w.rsa) ~b:w.rvb ~lead:2 ~tile:p.tile
         ~terms:(if c.rnk then 3 else 2)
         sb (A.layout b) w.rsb
  in
  check_loaders ~keyed launches;
  if keyed then equal (list string) [] (scale_passes launches);
  if scaled then begin
    cover_kernels (loaders ~rows:true ~scaled:true) launches;
    cover "the grouped float32 tiles" (ran "contract_rows_f32_")
  end
  else begin
    cover_layouts ~a:a_any ~b:b_any ~a_m:1 ~b_major:2 ~stage launches;
    let copied =
      List.exists (fun (t : Plan.traced) -> apart_copy t.kernel) launches
    in
    let l = A.layout b in
    let dense = A.Layout.strides (A.Layout.contiguous (A.Layout.shape l)) in
    cover "b's groups stepped, in place"
      ((not copied) && c.rg > 1 && A.Layout.stride l 1 = 2 * Iarray.get dense 1)
  end;
  let want =
    Array.init outputs (fun o ->
        let z = o / (rows * n) and i = o / n mod rows and j = o mod n in
        let id = id_of o in
        let init = Option.map (fun (_, v) -> v.(o)) init in
        if in_group c.rg id then
          let id = Int64.to_int id in
          associated_at ~ranges:p.ranges ~k ~init av
            (((z * rows) + i) * k)
            1 bv (b_index c z id 0 j)
            (if c.rnk then 1 else n)
        else associated_at ~ranges:p.ranges ~k:0 ~init av 0 1 bv 0 1)
  in
  (* Output [o] with term [kk] made +0, at float32, dst's dtype. *)
  let dropped o kk =
    let z = o / (rows * n) and i = o / n mod rows and j = o mod n in
    let init = Option.map (fun (_, v) -> v.(o)) init in
    let id = Int64.to_int (id_of o) in
    associated ~ranges:p.ranges ~k ~init (fun t ->
        if t = kk then (0., 0.)
        else (av.((((z * rows) + i) * k) + t), bv.(b_index c z id t j)))
  in
  if not scaled then
    cover_outputs ~da:w.rva ~db:w.rvb ~ranges:p.ranges ~outputs
      ~want:(fun o -> want.(o))
      ~dropped
      ~init:(fun o -> Option.fold ~none:0. ~some:(fun (_, v) -> v.(o)) init)
      ~terms:(fun o ->
        let z = o / (rows * n) and i = o / n mod rows and j = o mod n in
        let id = id_of o in
        if not (in_group c.rg id) then [||]
        else
          let id = Int64.to_int id in
          Array.init k (fun kk ->
              (av.((((z * rows) + i) * k) + kk), bv.(b_index c z id kk j))));
  let want = A.to_array (A.of_array D.Float32 (Iarray.of_list sy) want) in
  equal (array float_exact) want (A.to_array (Support.host dst))

let rconverted_of ?(rnk = false) ?rids ?rinit ?(rbview = Fresh_operands) ?rsa
    ?rsb rva rvb rbatch rm rn rk rg =
  {
    rv = rbits_of ~rnk ?rids ?rinit rbatch rm rn rk rg;
    rva;
    rvb;
    rbview;
    rsa;
    rsb;
  }

(* Mixed calls that reach each grouped tile, at each stage and layout of b, a
   widened operand, each view of the operands, and every float8 and float4
   code. *)
let rconverted_examples =
  let open D in
  List.concat_map
    (fun (rva, rvb) ->
      List.concat_map
        (fun rnk ->
          [
            rconverted_of ~rnk rva rvb 1 64 70 40 4;
            rconverted_of ~rnk rva rvb 1 20 64 33 8;
          ])
        [ false; true ])
    [
      (Any Float16, Any Float8_e4m3fn);
      (Any Bfloat16, Any Int8);
      (Any Float32, Any Int32);
    ]
  @ [
      rconverted_of (Any Float16) (Any Int4) 2 40 30 50 3;
      rconverted_of (Any Complex64) (Any Uint8) 1 30 20 70 2;
      rconverted_of (Any Float16) (Any Float8_e4m3fn) 1 20 64 2048 8;
      rconverted_of ~rids:Outside ~rinit:true (Any Int8) (Any Float32) 2 48 20
        50 3;
      rconverted_of ~rbview:Offset_operands (Any Float16) (Any Int4) 1 40 30 50
        3;
      rconverted_of ~rbview:Offset_operands (Any Int8) (Any Int64) 1 20 30 40 3;
      rconverted_of ~rbview:Broadcast_b (Any Float16) (Any Float8_e4m3fn) 2 30
        40 50 3;
      rconverted_of ~rbview:Stepped_b ~rnk:true (Any Bfloat16) (Any Int8) 1 30
        40 64 4;
      rconverted_of ~rbview:Stepped_b (Any Float32) (Any Uint16) 1 30 40 64 4;
      rconverted_of ~rbview:Offset_operands (Any Int16) (Any Float32) 1 30 40 50
        3;
      rconverted_of ~rbview:Broadcast_a (Any Float16) (Any Int8) 2 30 40 50 3;
      rconverted_of ~rbview:Stepped_a (Any Int8) (Any Float16) 1 40 30 50 4;
      rconverted_of ~rbview:Flipped (Any Float16) (Any Float8_e4m3fn) 1 40 30 50
        4;
      rconverted_of ~rbview:Flipped ~rnk:true (Any Float32) (Any Int16) 1 30 40
        64 3;
      rconverted_of ~rbview:Stepped_batch (Any Bfloat16) (Any Uint8) 2 30 40 50
        3;
      rconverted_of ~rbview:Stepped_groups (Any Float32) (Any Int32) 1 40 30 50
        4;
      rconverted_of (Any Float4_e2m1fn) (Any Float8_e5m2) 1 40 70 64 4;
    ]

(* nx_metal.mli: a Rows call whose casts pass the buffer is cut into pieces of
   its rows or its columns over every group: every output has the bits of the
   call uncut. *)
let law_rows_pieces (w, p) =
  let c = w.rv in
  let bk = backend () in
  let st = Random.State.make [| c.rseed |] in
  let sa, sb, sy = rows_shapes c in
  let a_view, b_view =
    match w.rbview with
    | Stepped_groups -> (`Fresh, `Step 1)
    | v -> views_of v ~a_m:1 ~b_major:2
  in
  let special = Int.max 64 (4 * c.rk) in
  let a_any, _ = laid_operand ~special st bk w.rva sa ~low:(-8) a_view in
  let b_any, _ = laid_operand ~special st bk w.rvb sb ~low:(-8) b_view in
  let (A.Any a) = a_any in
  let (A.Any b) = b_any in
  let ids = rows_ids st c in
  let outputs = List.fold_left ( * ) 1 sy in
  let init =
    if c.rinit then
      Some
        (fst
           (stored D.Float32 sy
              (drawn_values st ~name:"float32" ~low:(-8) ~special:16 outputs)))
    else None
  in
  let run () =
    let dst =
      Support.on bk (fst (stored D.Float32 sy (Array.make outputs nan)))
    in
    let got, launches =
      rows_call c ~a ~b ~ids ~dst
        ~init:(Option.map (fun x -> Support.on bk x) init)
    in
    equal answer A.Done got;
    (Support.elements (A.Any dst), launches)
  in
  let stage = stage_dtype ~scaled:false w.rva w.rvb in
  let cast_a = w.rva <> stage and cast_b = w.rvb <> stage in
  let sbytes = stage_bytes stage in
  let size cast x =
    if cast then sbytes * distinct_axes x (fun _ -> true) else 0
  in
  let total = size cast_a a_any + size cast_b b_any in
  let w_bytes =
    if cut_parts.(p) = 0 then Int.max 1 (total / 16)
    else Int.max 1 (total / cut_parts.(p))
  in
  let want, _ = run () in
  let got, launches = pieced w_bytes run in
  (* The outputs' axes: the batch's, two under split ids; a's rows, and its axis
     of 3; b's columns. A line of a is its terms, of b its groups' terms. *)
  let ka = if c.rsplit then 3 else 2 and kb = if c.rnk then 3 else 2 in
  let line cast x pick = if cast then sbytes * distinct_axes x pick else 0 in
  let line_a = line cast_a a_any (fun i -> i = ka) in
  let line_b = line cast_b b_any (fun i -> i = 1 || i = kb) in
  let axis extent a b = { Pc.extent; a; b; step = 1 } in
  let batch_axis extent =
    axis extent (cast_a && varies a_any 0) (cast_b && varies b_any 0)
  in
  let batch =
    if c.rview = Split_ids then [ batch_axis 2; batch_axis 2 ]
    else [ batch_axis c.rbatch ]
  in
  let rows =
    axis c.rm (cast_a && varies a_any 1) false
    :: (if c.rsplit then [ axis 3 (cast_a && varies a_any 2) false ] else [])
  in
  let cols =
    [ axis c.rn false (cast_b && varies b_any (if c.rnk then 2 else 3)) ]
  in
  let axes = Array.of_list (batch @ rows @ cols) in
  let boxes =
    (Pc.cut ~w:w_bytes ~batch:(List.length batch) ~rows:(List.length rows) axes
       ~a:line_a ~b:line_b)
      .boxes
  in
  let plans = contraction_plans launches in
  equal ~msg:"a contraction plan a piece" int
    (if c.rk = 0 || c.rg = 0 then 0 else Array.length boxes)
    plans;
  let cut = plans > 1 in
  let vector =
    List.exists
      (fun (t : Plan.traced) -> tile_prefix "contract_mv_" t.kernel)
      launches
  in
  let sizes = Array.map (Pc.bytes axes ~a:line_a ~b:line_b) boxes in
  let shares = band_shares ~vector ~w:w_bytes ~lines:(line_a, line_b) sizes in
  let bands = band_order ~width:sbytes shares launches in
  cover "a Rows call's b banded, its groups"
    (Array.exists (fun (_, y) -> y > 0) shares && c.rg > 1);
  cover "a Rows call's a banded" (Array.exists (fun (x, _) -> x > 0) shares);
  cover "a Rows call's band of several chunks" (bands.most_chunk > 0);
  cover "a grouped tile's twin"
    (List.exists
       (fun (t : Plan.traced) ->
         banded_twin t.kernel
         && String.starts_with ~prefix:"contract_rows_"
              (Nx_metal.Kernels.name t.kernel)
         && not (is_loader t.kernel))
       launches);
  cover "a cast, cut" (cast_a && (not cast_b) && cut);
  cover "b cast, cut" (cast_b && (not cast_a) && cut);
  cover "both cast, cut" (cast_a && cast_b && cut);
  cover "ids outside [0, g), cut" (c.rids = Outside && cut);
  equal (array string) want got

let rows_pieces_case =
  Gen.(with_pp (pp_pieced pp_rconverted) (pair rconverted_case (int_range 1 4)))

let rows_pieces_examples =
  let open D in
  [
    (rconverted_of (Any Float16) (Any Float8_e4m3fn) 1 64 70 40 4, 3);
    (rconverted_of ~rnk:true (Any Float8_e4m3fn) (Any Bfloat16) 1 64 70 40 4, 2);
    (rconverted_of (Any Int8) (Any Float8_e5m2) 2 48 100 50 3, 4);
    ( rconverted_of ~rids:Outside ~rinit:true (Any Int8) (Any Float32) 2 48 20
        50 3,
      2 );
    (rconverted_of (Any Float16) (Any Float8_e4m3fn) 1 4 2 9000 3, 0);
    (rconverted_of (Any Int8) (Any Float16) 1 2 3 9000 2, 0);
  ]

(* Rows calls of each kind of scale on a, on b and on both, blocks of 32 along
   k, one shared by the groups, at both layouts of b, a b narrower than a byte,
   and scales at an offset and stepped. *)
let rscaled_examples =
  let open D in
  [
    rconverted_of
      ~rsb:(sspec_of [ 0; 0; 2; 0 ])
      (Any Float16) (Any Float8_e4m3fn) 1 64 70 64 4;
    rconverted_of ~rnk:true
      ~rsb:(sspec_of [ 0; 0; 0; 2 ])
      (Any Float32) (Any Int8) 1 20 64 64 8;
    rconverted_of
      ~rsb:(sspec_of ~sexp:false ~sview:Permuted_scale [ 0; 1; 0; 0 ])
      (Any Float32) (Any Float32) 2 30 40 50 3;
    rconverted_of
      ~rsb:(sspec_of ~sexp:false ~sdt:(Any Bfloat16) [ 0; 0; 1; 0 ])
      (Any Bfloat16) (Any Int4) 1 40 30 64 3;
    rconverted_of
      ~rsb:(sspec_of [ 0; 0; 2; 0 ])
      (Any Float32) (Any Int8) 1 20 64 2048 8;
    rconverted_of ~rids:Outside ~rinit:true
      ~rsb:(sspec_of ~sexp:false [ 0; 0; 0; 1 ])
      (Any Float16) (Any Float16) 2 48 20 50 3;
    rconverted_of
      ~rsa:(sspec_of ~sview:Offset_scale [ 0; 0; 2 ])
      (Any Float8_e4m3fn) (Any Float16) 1 64 70 64 4;
    rconverted_of ~rnk:true
      ~rsa:(sspec_of ~sexp:false ~sview:Stepped_scale [ 0; 2; 0 ])
      ~rsb:
        (sspec_of ~sexp:false ~sdt:(Any Float16) ~sview:Offset_scale
           [ 0; 0; 0; 2 ])
      (Any Int8) (Any Float32) 2 20 40 64 3;
  ]
  @ scaled_loader_examples ~rank:4 (fun ~nk ~s a b ->
      [
        rconverted_of ~rnk:nk ~rsb:s a b 1 64 70 64 4;
        rconverted_of ~rnk:nk ~rsb:s a b 1 20 64 64 8;
      ])

(* Contracting segments *)

(* A Contracting call's operands: both float32, both float16 or both bfloat16;
   float8_e4m3fn by float16; float16 by bfloat16; complex64; or int8 by int32
   into int32. *)
type kops = K_f32 | K_f16 | K_bf16 | K_f8 | K_mixed | K_complex | K_int

let kops_name = function
  | K_f32 -> "float32"
  | K_f16 -> "float16"
  | K_bf16 -> "bfloat16"
  | K_f8 -> "float8_e4m3fn by float16"
  | K_mixed -> "float16 by bfloat16"
  | K_complex -> "complex64"
  | K_int -> "int8 by int32 into int32"

(* A call of [kbatch] x [km] x [kn] whose segmented pair has [kseg] terms,
   beside a pair of [kother] where it is not 0, the segmented one second where
   [ksecond]; a float call's a and b scaled where [ksa] and [ksb] say. *)
type kbits = {
  kops : kops;
  kbatch : int;
  km : int;
  kn : int;
  kseg : int;
  kother : int;
  ksecond : bool;
  kg : int;
  kids : drawn_ids;
  kview : ids_view;
  kinit : bool;
  ksa : sspec option;
  ksb : sspec option;
  kseed : int;
}

let pp_kbits ppf c =
  let scale name = function
    | None -> ""
    | Some sp -> Format.asprintf ", %s scaled by %a" name pp_sspec sp
  in
  Format.fprintf ppf
    "%d x %d x %d x %d%s in %d groups, %s, ids %s%s%s%s%s, seed %d" c.kbatch
    c.km c.kn c.kseg
    (if c.kother = 0 then ""
     else
       Printf.sprintf " beside a pair of %d %s" c.kother
         (if c.ksecond then "before" else "after"))
    c.kg (kops_name c.kops) (ids_name c.kids) (ids_view_name c.kview)
    (if c.kinit then ", init" else "")
    (scale "a" c.ksa) (scale "b" c.ksb) c.kseed

(* The contracted extent, and the segmented pair's index of term [q] of it, in C
   order of the pairs. *)
let kterms c = c.kseg * Int.max 1 c.kother

let kseg_index c q =
  if c.kother = 0 then q else if c.ksecond then q mod c.kseg else q / c.kother

(* The most outputs a draw takes, beside {!rows_terms} terms. *)
let k_outputs = 1 lsl 18

(* Groups of none, one, many, past the terms and past the prepass's chunks;
   terms past a range, a span's edge and a prepass part. *)
let kbits_case =
  let open Gen in
  let groups =
    frequency
      [
        (1, constant 0);
        (1, constant 1);
        (4, int_range 2 40);
        (1, of_list [ 384; 5000 ]);
      ]
  in
  let depth =
    frequency
      [
        (1, constant 0);
        (3, int_range 1 100);
        (2, int_range 101 2000);
        (1, of_list [ 33; 9000 ]);
      ]
  in
  let side = frequency [ (3, int_range 1 20); (1, int_range 21 70) ] in
  with_pp pp_kbits
    (let+ kops =
       of_list [ K_f32; K_f16; K_bf16; K_f8; K_mixed; K_complex; K_int ]
     and+ kbatch = frequency [ (3, constant 1); (1, int_range 2 3) ]
     and+ km = side
     and+ kn = side
     and+ kseg = depth
     and+ kother = frequency [ (3, constant 0); (1, int_range 2 5) ]
     and+ ksecond = bool
     and+ kg = groups
     and+ kids = of_list [ Each; Sorted; One; Two; Outside ]
     and+ kview =
       frequency
         [
           (4, constant Fresh_ids);
           (1, constant Offset_ids);
           (1, constant Stepped_ids);
           (1, constant Broadcast_ids);
         ]
     and+ kinit = bool
     and+ kseed = nat in
     let kbatch = if kview = Broadcast_ids then Int.max 2 kbatch else kbatch in
     let k = kseg * Int.max 1 kother in
     let per =
       Int.min
         (rows_terms / (kbatch * Int.max 1 k))
         (k_outputs / (kbatch * Int.max 1 kg))
     in
     let km = Int.max 1 (Int.min km per) in
     let kn = Int.max 1 (Int.min kn (per / km)) in
     {
       kops;
       kbatch;
       km;
       kn;
       kseg;
       kother;
       ksecond;
       kg;
       kids;
       kview;
       kinit;
       ksa = None;
       ksb = None;
       kseed;
     })

(* nx_metal.mli: a group of [terms] terms, of a call of [outputs], takes [S]
   ranges as a tile's: [T] the groups of 256 outputs, [G] 384, [S = 1] where [2
   T >= G], else [max 1 (min (min 64 ⌊r / 8⌋) ⌈G / T⌉)], [r = ⌈terms / 32⌉]. *)
let k_ranges ~outputs ~terms =
  let t = (outputs + 255) / 256 and g = 384 in
  let r = (terms + 31) / 32 in
  if t = 0 || 2 * t >= g then 1
  else Int.max 1 (Int.min (Int.min 64 (r / 8)) ((g + t - 1) / t))

let kfloat_dts = function
  | K_f32 -> ("float32", Stage D.Float32, Stage D.Float32)
  | K_f16 -> ("float16", Stage D.Float16, Stage D.Float16)
  | K_bf16 -> ("bfloat16", Stage D.Bfloat16, Stage D.Bfloat16)
  | K_f8 -> ("float16", Stage D.Float8_e4m3fn, Stage D.Float16)
  | K_mixed -> ("bfloat16", Stage D.Float16, Stage D.Bfloat16)
  | K_complex | K_int -> invalid_arg "kfloat_dts"

(* The contracted pairs' extents in order, and the segmented one's index. *)
let kpairs c =
  if c.kother = 0 then ([ c.kseg ], 0)
  else if c.ksecond then ([ c.kother; c.kseg ], 1)
  else ([ c.kseg; c.kother ], 0)

(* a's, b's and dst's shapes. *)
let kshapes c =
  let ks = fst (kpairs c) in
  ( c.kbatch :: c.km :: ks,
    (c.kbatch :: ks) @ [ c.kn ],
    [ c.kg; c.kbatch; c.km; c.kn ] )

(* Runs [c]: a of [batch; m; K...], b of [batch; K...; n], ids of [batch; kseg]
   viewed as [c.kview] says, dst of [g; batch; m; n]. Its launches. *)
let kcall ?sa ?sb c ~acc ~a ~b ~init ~dst ~ids =
  let bk = backend () in
  let ids = ids_viewed ~on:(Support.on bk) c.kview ~bt:c.kbatch ~m:c.kseg ids in
  let contracting = if c.kother = 0 then [ (2, 1) ] else [ (2, 1); (3, 2) ] in
  Plan.traced @@ fun () ->
  Nx_metal.contract
    ~batch:(Iarray.of_list [ (0, 0) ])
    ~contracting:(Iarray.of_list contracting)
    ~segments:
      (Some
         (Nx_kernel.Contracting { ids; groups = c.kg; pair = snd (kpairs c) }))
    ~acc ~dst ~init
    { Nx_kernel.array = A.Any (Support.on bk a); scale = sa }
    { Nx_kernel.array = A.Any (Support.on bk b); scale = sb }

(* Each batch element's groups' terms, in increasing order. *)
let kgroups c ids =
  let k = kterms c in
  Array.init c.kbatch (fun z ->
      let lists = Array.make c.kg [] in
      for q = k - 1 downto 0 do
        let id = ids.((z * c.kseg) + kseg_index c q) in
        if in_group c.kg id then
          let e = Int64.to_int id in
          lists.(e) <- q :: lists.(e)
      done;
      Array.map Array.of_list lists)

(* Output [o]'s group, batch element, row and column. *)
let k_at c o =
  let m = c.km and n = c.kn in
  (o / (c.kbatch * m * n), o / (m * n) mod c.kbatch, o / n mod m, o mod n)

(* The ranges of every output with terms, and whether two groups with terms take
   different ranges. *)
let k_spread ~ranges groups =
  let used = ref [] in
  Array.iter
    (Array.iter (fun qs ->
         let kg = Array.length qs in
         if kg > 0 then used := ranges kg :: !used))
    groups;
  (List.exists (fun s -> s > 1) !used, List.sort_uniq compare !used)

(* A float call: every output sums its group's terms alone, numbered
   from 0 in increasing order, in the stated association, then init; an
   output of a group without terms is init alone, or +0. A scaled
   operand's terms are its elements times their scales, as nx_metal.mli
   states. Covers what the draw reached. *)
(* Which outputs a Contracting call reached: finite over several ranges,
   NaN, infinite. *)
type kreached = { kfinite : bool; knan : bool; kinf : bool }

let kfloat_bits c =
  let bk = backend () in
  let name, Stage da, Stage db = kfloat_dts c.kops in
  let st = Random.State.make [| c.kseed |] in
  let special = Int.max 64 (4 * kterms c) in
  let draw n = drawn_values st ~name ~low:(-8) ~special n in
  let sa, sb, sy = kshapes c in
  let count = List.fold_left ( * ) 1 in
  let a, av = stored da sa (draw (count sa)) in
  let b, bv = stored db sb (draw (count sb)) in
  let scale_a = Option.map (fun sp -> drawn_scale st bk sp sa) c.ksa in
  let scale_b = Option.map (fun sp -> drawn_scale st bk sp sb) c.ksb in
  let scaled_by shape scale vs =
    match scale with
    | None -> vs
    | Some (_, f) -> Array.mapi (fun t v -> f (index_of shape t) v) vs
  in
  let av = scaled_by sa scale_a av and bv = scaled_by sb scale_b bv in
  let ids =
    ids_drawn st c.kids ~g:c.kg ~m:c.kseg ~bt:c.kbatch
      ~broadcast:(c.kview = Broadcast_ids)
  in
  let groups = kgroups c ids in
  let outputs = count sy in
  let count_of o =
    let e, z, _, _ = k_at c o in
    Array.length groups.(z).(e)
  in
  (* Half the inits of groups without terms are -0, which no addition may
     touch. *)
  let zeroed o v = if count_of o = 0 && Random.State.bool st then -0. else v in
  let init =
    if c.kinit then
      Some (stored D.Float32 sy (Array.mapi zeroed (draw outputs)))
    else None
  in
  let dst =
    Support.on bk (fst (stored D.Float32 sy (Array.make outputs nan)))
  in
  let got, launches =
    kcall ?sa:(Option.map fst scale_a) ?sb:(Option.map fst scale_b) c
      ~acc:(D.Any D.Float32) ~a ~b ~ids ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
  in
  equal answer A.Done got;
  let ranges terms = k_ranges ~outputs ~terms in
  let k = kterms c and m = c.km and n = c.kn in
  let want =
    Array.init outputs (fun o ->
        let e, z, i, j = k_at c o in
        let qs = groups.(z).(e) in
        let init = Option.map (fun (_, v) -> v.(o)) init in
        let kg = Array.length qs in
        let x = Array.map (fun q -> av.((((z * m) + i) * k) + q)) qs in
        let y = Array.map (fun q -> bv.((((z * k) + q) * n) + j)) qs in
        associated_at ~ranges:(ranges kg) ~k:kg ~init x 0 1 y 0 1)
  in
  equal (array float_exact) want (A.to_array (Support.host dst));
  let finite =
    Array.exists Fun.id
      (Array.mapi
         (fun o v -> ranges (count_of o) > 1 && Float.is_finite v)
         want)
  in
  let reached =
    {
      kfinite = finite;
      knan = Array.exists Float.is_nan want;
      kinf = Array.exists (fun v -> Float.abs v = infinity) want;
    }
  in
  let zero_init =
    match init with
    | Some (_, v) ->
        List.exists
          (fun o -> count_of o = 0 && Float.sign_bit v.(o) && v.(o) = 0.)
          (List.init outputs Fun.id)
    | None -> false
  in
  (k_spread ~ranges groups, zero_init, reached, launches)

(* A complex64 call: as {!kfloat_bits}, each part the stated association of its
   real form's 2 K_g terms. *)
let kcomplex_bits c =
  let bk = backend () in
  let st = Random.State.make [| c.kseed |] in
  let special = Int.max 64 (8 * kterms c) in
  let draw n = drawn_values st ~name:"float32" ~low:(-8) ~special (2 * n) in
  let sa, sb, sy = kshapes c in
  let count = List.fold_left ( * ) 1 in
  let a, av = stored_complex sa (draw (count sa)) in
  let b, bv = stored_complex sb (draw (count sb)) in
  let ids =
    ids_drawn st c.kids ~g:c.kg ~m:c.kseg ~bt:c.kbatch
      ~broadcast:(c.kview = Broadcast_ids)
  in
  let groups = kgroups c ids in
  let outputs = count sy in
  let init =
    if c.kinit then Some (stored_complex sy (draw outputs)) else None
  in
  let dst = A.create (S.device ()) D.Complex64 (Iarray.of_list sy) in
  let got, launches =
    kcall c ~acc:(D.Any D.Complex64) ~a ~b ~ids ~dst
      ~init:(Option.map (fun (x, _) -> Support.on bk x) init)
  in
  equal answer A.Done got;
  let ranges terms = k_ranges ~outputs:(2 * outputs) ~terms:(2 * terms) in
  let k = kterms c and m = c.km and n = c.kn in
  let part o im =
    let e, z, i, j = k_at c o in
    let qs = groups.(z).(e) in
    let kg = Array.length qs in
    let init =
      Option.map
        (fun (_, v) -> if im then v.(o).Complex.im else v.(o).Complex.re)
        init
    in
    (* The real form's terms 2p and 2p + 1 of the group's term p. *)
    let x = Array.make (2 * kg) 0. and y = Array.make (2 * kg) 0. in
    Array.iteri
      (fun p q ->
        let u = av.((((z * m) + i) * k) + q)
        and v = bv.((((z * k) + q) * n) + j) in
        x.(2 * p) <- u.Complex.re;
        x.((2 * p) + 1) <- u.im;
        y.(2 * p) <- (if im then v.Complex.im else v.re);
        y.((2 * p) + 1) <- (if im then v.re else -.v.im))
      qs;
    associated_at ~ranges:(ranges kg) ~k:(2 * kg) ~init x 0 1 y 0 1
  in
  let got = A.to_array (Support.host dst) in
  let re = Array.init outputs (fun o -> part o false) in
  let im = Array.init outputs (fun o -> part o true) in
  equal ~msg:"real parts" (array float_exact) re
    (Array.map (fun (x : Complex.t) -> x.re) got);
  equal ~msg:"imaginary parts" (array float_exact) im
    (Array.map (fun (x : Complex.t) -> x.im) got);
  let count_of o =
    let e, z, _, _ = k_at c o in
    Array.length groups.(z).(e)
  in
  let finite =
    List.exists
      (fun o ->
        ranges (count_of o) > 1
        && Float.is_finite re.(o)
        && Float.is_finite im.(o))
      (List.init outputs Fun.id)
  in
  let parts = Array.append re im in
  let reached =
    {
      kfinite = finite;
      knan = Array.exists Float.is_nan parts;
      kinf = Array.exists (fun v -> Float.abs v = infinity) parts;
    }
  in
  (k_spread ~ranges groups, false, reached, launches)

(* An integer call into int32: every output is the exact sum of its group's
   products, plus init, modulo 2^32; one of a group without terms is init alone,
   or 0. *)
let kint_bits c =
  let bk = backend () in
  let st = Random.State.make [| c.kseed |] in
  let acc = D.Any D.Int32 in
  let sa, sb, sy = kshapes c in
  let A.Any a, av = int_operand st (D.Any D.Int8) sa ~acc in
  let A.Any b, bv = int_operand st (D.Any D.Int32) sb ~acc in
  let ids =
    ids_drawn st c.kids ~g:c.kg ~m:c.kseg ~bt:c.kbatch
      ~broadcast:(c.kview = Broadcast_ids)
  in
  let groups = kgroups c ids in
  let count = List.fold_left ( * ) 1 in
  let outputs = count sy in
  let init = if c.kinit then Some (int_operand st acc sy ~acc) else None in
  let dst =
    Support.on bk
      (A.of_array D.Int32 (Iarray.of_list sy) (Array.make outputs (-1l)))
  in
  let got, launches =
    kcall c ~acc ~a ~b ~ids ~dst
      ~init:
        (Option.map
           (fun (A.Any x, _) -> A.expect D.Int32 (A.Any (Support.on bk x)))
           init)
  in
  equal answer A.Done got;
  let k = kterms c and m = c.km and n = c.kn in
  let sums =
    Array.init outputs (fun o ->
        let e, z, i, j = k_at c o in
        let s =
          Array.fold_left
            (fun s q ->
              Int64.add s
                (Int64.mul
                   av.((((z * m) + i) * k) + q)
                   bv.((((z * k) + q) * n) + j)))
            0L
            groups.(z).(e)
        in
        let s =
          match init with None -> s | Some (_, v) -> Int64.add s v.(o)
        in
        Int64.to_int32 s)
  in
  equal (array int32) sums (A.to_array (Support.host dst));
  ((false, []), false, { kfinite = false; knan = false; kinf = false }, launches)

let kbits_run c =
  match c.kops with
  | K_int -> kint_bits c
  | K_complex -> kcomplex_bits c
  | K_f32 | K_f16 | K_bf16 | K_f8 | K_mixed -> kfloat_bits c

(* nx_kernel.mli and nx_metal.mli: a Contracting call's output sums its group's
   terms alone, numbered from 0 in their order, with the bits of the stated
   association of its own count, whatever the other indices' groups; a group
   without terms has no products: init alone, or +0. At each dtype it computes,
   ranges with finite outputs, ids in every arrangement, g = 0, g past the
   terms, past a chunk of the prepass's scan of counts and past 2^16, a group of
   several parts past the scan's first chunk, NaN and infinite outputs, and
   either pair of two segmented; integers exactly. *)
let law_kbits c =
  let k = kterms c in
  cover "float16" (c.kops = K_f16);
  cover "bfloat16" (c.kops = K_bf16);
  cover "float8_e4m3fn by float16" (c.kops = K_f8);
  cover "float16 by bfloat16" (c.kops = K_mixed);
  cover "complex64" (c.kops = K_complex);
  cover "integers" (c.kops = K_int);
  cover "no group" (c.kg = 0);
  cover "one group" (c.kg = 1);
  cover "g past the terms" (c.kg > k && k > 0);
  cover "g past a chunk of the prepass's scan" (c.kg > 2048);
  cover "g past 2^16" (c.kg > 1 lsl 16);
  cover "ids outside [0, g)" (c.kids = Outside);
  cover "sorted ids" (c.kids = Sorted);
  cover "empty groups" ((c.kids = One || c.kids = Two) && c.kg > 2);
  cover "two pairs, the first segmented" (c.kother > 0 && not c.ksecond);
  cover "two pairs, the second segmented" (c.kother > 0 && c.ksecond);
  cover "terms past a part of the prepass" (k > 4096);
  (* nx_metal.mli: parts of at least 4096 indices; the scan takes 2048 counts, a
     group's parts in turn, a chunk. *)
  let parts = (k + 4095) / 4096 in
  cover "a group of several parts past the scan's first chunk"
    (parts > 1 && c.kg * parts > 2048);
  cover "an init" c.kinit;
  cover "a batch" (c.kbatch > 1);
  cover "no product" (k = 0 && c.kg > 0);
  cover "ids at an offset" (c.kview = Offset_ids);
  cover "stepped ids" (c.kview = Stepped_ids);
  cover "ids broadcast along the batch" (c.kview = Broadcast_ids);
  let (several, spread), zero_init, r, _ = kbits_run c in
  cover "several ranges" several;
  cover "a finite output over several ranges" r.kfinite;
  cover "a NaN output" r.knan;
  cover "an infinite output" r.kinf;
  cover "groups of one call in different ranges" (List.length spread > 1);
  cover "a group without terms under an init of -0" zero_init

let kbits_of ?(kother = 0) ?(ksecond = false) ?(kids = Each)
    ?(kview = Fresh_ids) ?(kinit = false) ?ksa ?ksb kops kbatch km kn kseg kg =
  {
    kops;
    kbatch;
    km;
    kn;
    kseg;
    kother;
    ksecond;
    kg;
    kids;
    kview;
    kinit;
    ksa;
    ksb;
    kseed = 11;
  }

(* Calls that reach each dtype, several ranges and finite outputs over them,
   groups in different ranges, g = 0 and 1, g past the terms, past a chunk of
   the prepass's scan and past 2^16, terms past a part of it, each arrangement
   and view of the ids, either pair of two, no product, and a group without
   terms under an init of -0. *)
let kbits_examples =
  [
    kbits_of K_f32 1 16 16 2048 2;
    kbits_of K_f16 1 20 9 1500 5;
    kbits_of K_bf16 2 9 20 700 3;
    kbits_of K_f8 1 20 20 1000 3;
    kbits_of K_mixed 1 7 30 800 2;
    kbits_of K_complex 1 10 10 1000 2;
    kbits_of K_int 2 10 10 300 3;
    kbits_of K_f32 1 4 4 40 0;
    kbits_of K_f32 1 30 30 50 1;
    kbits_of K_f32 1 1 1 600 5000;
    kbits_of K_f32 1 1 1 64 100_000;
    kbits_of K_f32 1 1 1 9000 1024;
    kbits_of K_f32 1 3 3 9000 4;
    kbits_of K_f32 1 8 8 30 60;
    kbits_of ~kids:Outside ~kinit:true K_f32 1 10 10 600 4;
    kbits_of ~kids:Sorted K_f16 1 10 10 600 4;
    kbits_of ~kids:Two ~kinit:true K_f32 1 10 10 600 6;
    kbits_of ~kids:One K_complex 1 5 5 300 4;
    kbits_of ~kother:3 K_f32 1 9 9 200 3;
    kbits_of ~kother:4 ~ksecond:true K_bf16 1 9 9 200 3;
    kbits_of ~kother:2 ~ksecond:true K_int 1 9 9 100 3;
    kbits_of ~kinit:true K_f32 2 6 6 0 3;
    kbits_of ~kview:Offset_ids K_f32 1 10 10 300 3;
    kbits_of ~kview:Stepped_ids K_f16 1 10 10 300 3;
    kbits_of ~kview:Broadcast_ids K_f32 2 10 10 300 3;
  ]

(* Float calls of a scaled a, b or both. *)
let kscaled_case =
  Gen.(
    with_pp pp_kbits
      (let* c = kbits_case in
       let rank = if c.kother = 0 then 3 else 4 in
       let+ which = int_range 1 3
       and+ kops = of_list [ K_f32; K_f16; K_bf16; K_f8; K_mixed ]
       and+ sa = sspec_case rank
       and+ sb = sspec_case rank in
       {
         c with
         kops;
         ksa = (if which land 1 <> 0 then Some sa else None);
         ksb = (if which land 2 <> 0 then Some sb else None);
       }))

(* nx_kernel.mli and nx_metal.mli: a scaled operand's term in a Contracting call
   is its element times its scale, rounded once into float32, in each group's
   sum alone. *)
let law_kscaled c =
  cover "a scaled" (Option.is_some c.ksa && Option.is_none c.ksb);
  cover "b scaled" (Option.is_none c.ksa && Option.is_some c.ksb);
  cover "both scaled" (Option.is_some c.ksa && Option.is_some c.ksb);
  cover "an Exponent scale"
    (List.exists
       (fun (s : sspec) -> s.sexp)
       (List.filter_map Fun.id [ c.ksa; c.ksb ]));
  cover "a Value scale"
    (List.exists
       (fun (s : sspec) -> not s.sexp)
       (List.filter_map Fun.id [ c.ksa; c.ksb ]));
  let (several, _), _, r, _ = kfloat_bits c in
  cover "several ranges" several;
  cover "a finite output over several ranges" r.kfinite;
  cover "a NaN output" r.knan;
  cover "an infinite output" r.kinf

(* Calls that reach each kind of scale on a, on b and on both, and several
   ranges. *)
let kscaled_examples =
  [
    kbits_of ~ksa:(sspec_of [ 0; 0; 2 ]) K_f32 1 16 16 2048 2;
    kbits_of
      ~ksb:(sspec_of ~sexp:false ~sdt:(D.Any D.Float16) [ 0; 2; 1 ])
      K_f8 1 8 8 600 3;
    kbits_of
      ~ksa:(sspec_of ~sexp:false [ 0; 1; 2 ])
      ~ksb:(sspec_of [ 1; 2; 0 ])
      K_bf16 2 6 6 300 3;
  ]

type kwindowed = { kc : kbits; kallowed : float }

let pp_kwindowed ppf w =
  Format.fprintf ppf "%a, launches of at most %gs" pp_kbits w.kc w.kallowed

(* Calls of few groups and terms, whose prepass fits the least budget: its
   listing takes a step of a dependent read and 32 shuffles for each 32 terms of
   a part. *)
let kwindowed_case =
  Gen.(
    with_pp pp_kwindowed
      (let+ kc = kbits_case and+ kallowed = of_list [ 3e-5; 1e-4 ] in
       let kc =
         {
           kc with
           kg = Int.min kc.kg 40;
           kseg = Int.min kc.kseg 48;
           kother = 0;
         }
       in
       { kc; kallowed }))

(* nx_metal.mli: no bit depends on the launches. Under budgets that cut a
   Contracting call's outputs into windows, or their ranges into chunks, every
   output keeps its bits. *)
let law_k_window_bits w =
  let module K = Nx_metal.Kernels in
  let _, _, _, launches = bounded w.kallowed (fun () -> kbits_run w.kc) in
  let sums =
    List.filter (fun (t : Plan.traced) -> t.kernel = K.Contract_simt) launches
  in
  cover "several windows" (List.exists (fun t -> word t K.simt_first > 0) sums);
  cover "chunks of k" (List.exists (fun t -> word t K.simt_length > 0) sums)

(* Calls that a budget cuts into several windows, and into chunks of k: a group
   of 2^20 terms, 2^19 complex ones, in 64 ranges, whose threads take longer
   than 1 ms, past its prepass's launches. *)
let kwindowed_examples =
  [
    { kc = kbits_of K_f32 2 64 64 64 3; kallowed = 3e-5 };
    { kc = kbits_of K_int 2 64 64 64 3; kallowed = 3e-5 };
    { kc = kbits_of ~kids:One K_f32 1 2 2 (1 lsl 20) 1; kallowed = 1e-3 };
    { kc = kbits_of ~kids:One K_complex 1 1 2 (1 lsl 19) 1; kallowed = 1e-3 };
    { kc = kbits_of ~kids:One K_int 1 2 2 (1 lsl 20) 1; kallowed = 1e-3 };
  ]

(* Contraction windows *)

type windowed = { wc : bits; cut : int }

let pp_windowed ppf w =
  Format.fprintf ppf "%a, launches of 1/%d of its time" pp_bits w.wc w.cut

let windowed_case =
  Gen.(
    with_pp pp_windowed
      (let+ wc = bits_case and+ cut = of_list [ 2; 8; 64; 300 ] in
       let exchanged = C.exchanges ~m:wc.bm ~n:wc.bn in
       { wc = { wc with at = false; bnk = wc.bnk || exchanged }; cut }))

(* The least budget the window law draws: at its extents, a launch of a tile
   over 32 terms and a finish of a row each take under 1e-5 s past their
   dispatch. Its operands need no copy, whose launches of a few elements take
   more: a is not strided, and b is read n x k where it gives the tiles'
   rows. *)
let window_floor = Plan.dispatch +. 1e-5

(* nx_metal.mli: no bit depends on the launches. Under a budget of a fraction of
   the call's time, its tiles run in several windows, or in chunks of k that
   each resume the sums before them, and every output keeps the bits of the
   stated association. *)
let law_window_bits w =
  let lib = library_plan w.wc in
  let allowed =
    Float.min Plan.budget
      (Float.max window_floor (lib.seconds /. Float.of_int w.cut))
  in
  let tiles =
    List.filter tile_launch (bounded allowed (fun () -> contracted_bits w.wc))
  in
  let first t = word t Nx_metal.Kernels.contract_first in
  let windows = List.sort_uniq Int.compare (List.map first tiles) in
  let chunked =
    List.exists (fun t -> word t Nx_metal.Kernels.contract_length > 0) tiles
  in
  cover "several windows" (List.length windows > 1);
  cover "chunks" chunked;
  cover "whole ranges" (tiles <> [] && not chunked)

(* Calls that a budget cuts into windows of whole ranges, and into chunks at
   each staged dtype. *)
let windowed_examples =
  [
    { wc = bits_of 1 4096 256 1; cut = 16 };
    { wc = bits_of 1 16 64 4096; cut = 300 };
    { wc = bits_of ~stage:1 ~bnk:true ~binit:true 1 16 64 4096; cut = 300 };
    { wc = bits_of ~stage:2 ~bdst:1 1 16 64 4096; cut = 300 };
    { wc = bits_of 1 16 129 4096; cut = 300 };
    { wc = bits_of ~stage:1 1 40 47 4096; cut = 300 };
  ]

(* Contraction declines *)

let device_bytes x =
  let b = A.buffer x in
  let n = Rig.Buffer.length b in
  let s = Bytes.create n in
  Rig.Buffer.blit_to_bytes b 0 s 0 n;
  Bytes.to_string s

let operand ?scale x = { Nx_kernel.array = A.Any x; scale }

(* [contract] of [a] by [b] into a drawn [dst] of [4; 4] unless given, with an
   [init] of [dst]'s dtype where [init], and whether [dst] kept its bytes. *)
let declined_call ?(acc = D.Any D.Float32) ?(segments = None) ?(init = false)
    ?(batch = [||]) ?(contracting = [| (1, 0) |]) ?dst a b =
  let st = Random.State.make [| 3 |] in
  let (A.Any dst) =
    match dst with Some d -> d | None -> A.Any (drawn st D.Float32 [| 4; 4 |])
  in
  let before = device_bytes dst in
  let init =
    if init then
      Some (A.create (A.device dst) (A.dtype dst) (L.shape (A.layout dst)))
    else None
  in
  let got =
    Nx_metal.contract ~batch:(Iarray.of_array batch)
      ~contracting:(Iarray.of_array contracting)
      ~segments ~acc ~dst ~init a b
  in
  (got, String.equal before (device_bytes dst))

let fresh dt s = drawn (Random.State.make [| 5 |]) dt s
let f32 s = operand (fresh D.Float32 s)

(* nx_metal.mli: the calls a contraction declines, each before it writes dst: a
   float64 or complex128 value anywhere, a scale's among them, an accumulator or
   operands it does not compute in, and a scale with an integer or complex
   accumulator. Each row makes its arrays as it runs: the suite's declaration
   opens no device, so that it declares its tests on a machine without Metal and
   the process that runs no test holds none. *)
let decline_cases =
  let f64 s = operand (fresh D.Float64 s) in
  let exponent () = Nx_kernel.Exponent (fresh D.Uint8 [| 1; 1 |]) in
  let wide () = Nx_kernel.Value (fresh D.Float64 [| 1; 1 |]) in
  [
    ( "float64 operands",
      fun () ->
        declined_call ~acc:(D.Any D.Float64)
          ~dst:(A.Any (fresh D.Float64 [| 4; 4 |]))
          (f64 [| 4; 8 |])
          (f64 [| 8; 4 |]) );
    ( "a float64 accumulator",
      fun () ->
        declined_call ~acc:(D.Any D.Float64) (f32 [| 4; 8 |]) (f32 [| 8; 4 |])
    );
    ( "a float64 operand",
      fun () -> declined_call (f32 [| 4; 8 |]) (f64 [| 8; 4 |]) );
    ( "a float64 dst",
      fun () ->
        declined_call
          ~dst:(A.Any (fresh D.Float64 [| 4; 4 |]))
          (f32 [| 4; 8 |])
          (f32 [| 8; 4 |]) );
    ( "a float64 dst and init",
      fun () ->
        declined_call ~init:true
          ~dst:(A.Any (fresh D.Float64 [| 4; 4 |]))
          (f32 [| 4; 8 |])
          (f32 [| 8; 4 |]) );
    ( "complex128 operands",
      fun () ->
        declined_call ~acc:(D.Any D.Complex128)
          ~dst:(A.Any (fresh D.Complex128 [| 4; 4 |]))
          (operand (fresh D.Complex128 [| 4; 8 |]))
          (operand (fresh D.Complex128 [| 8; 4 |])) );
    ( "a complex64 accumulator with a scale",
      fun () ->
        declined_call ~acc:(D.Any D.Complex64)
          ~dst:(A.Any (fresh D.Complex64 [| 4; 4 |]))
          (operand ~scale:(exponent ()) (fresh D.Complex64 [| 4; 8 |]))
          (operand (fresh D.Complex64 [| 8; 4 |])) );
    ( "float32 operands into complex64",
      fun () ->
        declined_call ~acc:(D.Any D.Complex64)
          ~dst:(A.Any (fresh D.Complex64 [| 4; 4 |]))
          (f32 [| 4; 8 |])
          (f32 [| 8; 4 |]) );
    ( "an int32 accumulator with a scale",
      fun () ->
        declined_call ~acc:(D.Any D.Int32)
          ~dst:(A.Any (fresh D.Int32 [| 4; 4 |]))
          (operand ~scale:(exponent ()) (fresh D.Int8 [| 4; 8 |]))
          (operand (fresh D.Int8 [| 8; 4 |])) );
    ( "a float64 operand into int32",
      fun () ->
        declined_call ~acc:(D.Any D.Int32)
          ~dst:(A.Any (fresh D.Int32 [| 4; 4 |]))
          (operand (fresh D.Int32 [| 4; 8 |]))
          (f64 [| 8; 4 |]) );
    ( "a float64 scale",
      fun () ->
        declined_call
          (f32 [| 4; 8 |])
          (operand ~scale:(wide ()) (fresh D.Float32 [| 8; 4 |])) );
  ]

let test_declines (_, call) =
  ignore (backend ());
  let got, kept = call () in
  equal answer A.Declined got;
  equal ~msg:"dst kept its bytes" bool true kept

(* nx_metal.mli: a call whose launches take past 10 s together by the estimate
   runs as several parts, each within 10 s, and one within as one. The calls
   broadcast a row of k terms to every row of a and of b, read n x k, so that a
   long call takes little memory: 2048 x 2048 outputs, k where the tiles'
   estimate passes 10 s, and where it is 0.1 s short of it, which the finish
   does not take. *)
let test_part_budget () =
  let bk = backend () in
  let d = S.device () in
  let m = 2048 in
  let at k =
    C.plan
      {
        C.batch = 1;
        m;
        n = m;
        k;
        stage = C.F32;
        nk = true;
        complex = false;
        source = 0;
      }
  in
  let tiles_seconds k = (Option.get (at k)).seconds in
  (* The least multiple of 32 terms whose tiles take past [limit]. *)
  let rec past limit lo hi =
    if hi - lo <= 32 then hi
    else
      let mid = (lo + hi) / 64 * 32 in
      if tiles_seconds mid > limit then past limit lo mid else past limit mid hi
  in
  let past limit = past limit 32 (1 lsl 24) in
  let within = past (Plan.part_budget -. 0.1) - 32 in
  let call k =
    let row = Support.on bk (fresh D.Float32 [| 1; k |]) in
    let wide x =
      Option.get (A.move (M.Broadcast (Iarray.of_list [ m; k ])) x)
    in
    let dst = A.create d D.Float32 (Iarray.of_list [ m; m ]) in
    let x = wide row in
    C.call
      ~ops:[| A.Any dst; A.Any x; A.Any x |]
      ~batch:(Iarray.of_list [])
      ~contracting:(Iarray.of_list [ (1, 1) ])
      ~segments:None ~acc:(D.Any D.Float32) ~dst:(A.Any dst) ~init:None
      (operand x) (operand x)
  in
  let seconds (c : Plan.t) =
    Array.fold_left
      (fun s (l : Plan.launch) ->
        s
        +. Plan.seconds
             ~threads:(Nx_metal.Kernels.threads l.kernel)
             ~groups:l.groups ~resident:l.resident l.cost)
      0. c.launches
  in
  let part_seconds (p : Plan.part) =
    Array.fold_left
      (fun s (l : Plan.launch) ->
        s
        +. Plan.seconds
             ~threads:(Nx_metal.Kernels.threads l.kernel)
             ~groups:l.groups ~resident:l.resident l.cost)
      0. p.run
  in
  (match call within with
  | Ok (C.Planned (Some c)) ->
      at_most ~msg:"a call within" float_exact ~than:Plan.part_budget
        (seconds c);
      equal ~msg:"a call within, its parts" int 1 (Array.length c.parts)
  | Ok (C.Planned None | C.Recast _) -> fail "no launch"
  | Error r ->
      failf "a call within 10 s answered %a" Nx_array_support.pp_answer r);
  match call (past Plan.part_budget) with
  | Ok (C.Planned (Some c)) ->
      greater ~msg:"a call past 10 s, its parts" int ~than:1
        (Array.length c.parts);
      Array.iter
        (fun (p : Plan.part) ->
          at_most ~msg:"a part" float_exact ~than:Plan.part_budget
            (part_seconds p))
        c.parts
  | Ok (C.Planned None | C.Recast _) -> fail "no launch"
  | Error r -> failf "a call past 10 s answered %a" Nx_array_support.pp_answer r

(* Parts *)

type cut = Launches of int | Seconds of float

let pp_cut ppf = function
  | Launches n -> Format.fprintf ppf "parts of %d launches" n
  | Seconds t -> Format.fprintf ppf "parts of %g s" t

(* Bounds that cut most calls of several launches: one to three launches a part,
   or a few launches' dispatches. *)
let cut =
  let open Gen in
  with_pp pp_cut
    (frequency
       [
         (3, map (fun n -> Launches n) (int_range 1 3));
         ( 1,
           map
             (fun k -> Seconds (Float.of_int k *. Plan.dispatch))
             (int_range 1 4) );
       ])

let parted c f =
  match c with
  | Launches n -> S.parted ~launches:n f
  | Seconds t -> S.parted ~seconds:t f

(* A plan's launches: each of a kernel of 2, 4 or 32 words, over up to 4096
   groups, writing word 0 of one of 4 operands' slots or of the workspace. *)
type planned = {
  launches : (int * int * int) list;
  most : int;
  seconds : float;
  peak : int;
}

let pp_planned ppf c =
  Format.fprintf ppf "%d launches, parts of %d launches and %g s, peak %d"
    (List.length c.launches) c.most c.seconds c.peak

let planned_parts =
  let open Gen in
  let launch =
    let+ k = int_range 0 2
    and+ g = int_range 1 4096
    and+ slot = int_range 0 4 in
    (k, g, slot)
  in
  with_pp pp_planned
    (let+ launches = list ~size:(int_range 1 40) launch
     and+ most = int_range 1 8
     and+ seconds = float_range 2.1e-6 5e-4
     and+ peak = frequency [ (1, constant 0); (1, int_range 1 4096) ] in
     { launches; most; seconds; peak })

let part_kernels =
  [| Nx_metal.Kernels.Zero; Nx_metal.Kernels.Copy_run; Nx_metal.Kernels.Copy |]

let plan_of c =
  let b = Plan.make () in
  if c.peak > 0 then ignore (Plan.take b c.peak);
  List.iter
    (fun (k, g, slot) ->
      let kernel = part_kernels.(k) in
      let slot = if slot = 4 then Plan.workspace b else Plan.slot b slot in
      Plan.launch b kernel ~groups:(g, 1, 1)
        ~cost:(cost ~bytes:(4096 * g) ())
        (Plan.words kernel)
        [| { Plan.word = 0; slot; write = true } |])
    c.launches;
  Plan.finish b

let launch_time (l : Plan.launch) =
  Plan.seconds
    ~threads:(Nx_metal.Kernels.threads l.kernel)
    ~groups:l.groups ~resident:l.resident l.cost

(* Calls that reach law_cuts' rarer covers on every run: launches of 4096
   groups, each past half the seconds bound, cut by it, then launches of one
   group, which a later part's bytes cut; and one part of 150 launches of 4096
   groups, past commit_after. *)
let cuts_examples =
  let heavy = (0, 4096, 0) and light = (0, 1, 1) in
  let one = { launches = [ heavy ]; most = 1; seconds = 1.; peak = 0 } in
  let t = launch_time (plan_of one).launches.(0) in
  [
    {
      launches = [ heavy; heavy; heavy ] @ List.init 8 (fun _ -> light);
      most = 8;
      seconds = 1.5 *. t;
      peak = 0;
    };
    {
      launches = List.init 150 (fun _ -> heavy);
      most = 200;
      seconds = Plan.part_budget;
      peak = 0;
    };
  ]

(* plan.mli's bytes of a part's launches: their parameters, each launch's
   rounded up to 256 bytes, and their values, [32 + 16 w] bytes a launch of [w]
   words. *)
let run_bytes run =
  Array.fold_left
    (fun (a, v) (l : Plan.launch) ->
      let w = Bytes.length l.words in
      (a + ((w + 255) / 256 * 256), v + 32 + (2 * w)))
    (0, 0) run

let run_time run = Array.fold_left (fun t l -> t +. launch_time l) 0. run

(* plan.mli: [finish] cuts a call's launches in order into parts, each as long
   as its bounds allow, each but the first within the first's bytes, each
   committed where its seconds pass commit_after; a part numbers the slots its
   launches name in the order they name them; a call of several parts takes a
   workspace of its own. *)
let law_cuts c =
  let call =
    S.parted ~launches:c.most ~seconds:c.seconds (fun () -> plan_of c)
  in
  let parts = Array.to_list call.parts in
  cover "one part" (List.length parts = 1);
  cover "several parts" (List.length parts > 1);
  cover "a part cut by its launches"
    (List.exists (fun (p : Plan.part) -> Array.length p.run = c.most) parts);
  equal ~msg:"the parts' launches are the call's, in order" bool true
    (List.concat_map (fun (p : Plan.part) -> Array.to_list p.run) parts
    |> List.for_all2 ( == ) (Array.to_list call.launches));
  let first_bytes = run_bytes (List.hd parts).run in
  let by_seconds = ref false and by_bytes = ref false and commits = ref false in
  List.iteri
    (fun i (p : Plan.part) ->
      let n = Array.length p.run in
      let msg = Printf.sprintf "part %d" i in
      at_most ~msg:(msg ^ "'s launches") int ~than:c.most n;
      if p.commits then commits := true;
      equal
        ~msg:(msg ^ " commits past commit_after")
        bool
        (run_time p.run > Plan.commit_after)
        p.commits;
      if n > 1 then begin
        at_most ~msg:(msg ^ "'s seconds") float_exact ~than:c.seconds
          (run_time p.run);
        if i > 0 then begin
          let a, v = run_bytes p.run in
          at_most ~msg:(msg ^ "'s parameters") int ~than:(fst first_bytes) a;
          at_most ~msg:(msg ^ "'s values") int ~than:(snd first_bytes) v
        end
      end;
      (* Each part takes as many launches as its bounds allow. *)
      (match List.nth_opt parts (i + 1) with
      | None -> ()
      | Some (q : Plan.part) ->
          let longer = Array.append p.run [| q.run.(0) |] in
          let a, v = run_bytes longer in
          let launches = n + 1 > c.most
          and seconds = run_time longer > c.seconds
          and bytes = i > 0 && (a > fst first_bytes || v > snd first_bytes) in
          if seconds && not launches then by_seconds := true;
          if bytes && not (launches || seconds) then by_bytes := true;
          equal
            ~msg:(msg ^ " ends where a bound stops it")
            bool true
            (launches || seconds || bytes));
      let named = ref [] in
      Array.iter
        (fun (l : Plan.launch) ->
          Array.iter
            (fun (r : Plan.ref) ->
              if not (List.mem r.slot !named) then named := r.slot :: !named)
            l.refs)
        p.run;
      equal ~msg:(msg ^ "'s slots") (array int)
        (Array.of_list (List.rev !named))
        (if List.length parts = 1 then
           Array.init (Array.length call.slots) Fun.id
         else p.binds);
      Array.iteri
        (fun j k -> equal ~msg:(msg ^ "'s local slot") int j p.local.(k))
        p.binds;
      equal ~msg:(msg ^ " matches its key") bool true
        (Plan.matches (Plan.key p) p))
    parts;
  let ws =
    Array.exists
      (fun (s : Plan.slot) ->
        match s with Workspace | Fresh _ -> true | _ -> false)
      call.slots
  in
  let fresh =
    Array.exists
      (fun (s : Plan.slot) -> match s with Fresh _ -> true | _ -> false)
      call.slots
  in
  cover "a part that commits" !commits;
  cover "a part cut by its seconds" !by_seconds;
  cover "a later part cut by the first's bytes" !by_bytes;
  cover "a workspace over several parts" (ws && List.length parts > 1);
  equal ~msg:"a workspace of its own over several parts" bool
    (ws && List.length parts > 1)
    fresh

(* Calls at nx.metal's own part bounds: 300 to 400 launches, each estimated past
   80% of a launch's budget, which their seconds cut; or 10^5 and up to 1000
   more launches of a few bytes, which their count cuts. Each launch's kernel,
   of 2, 4 or 32 words, and slot are drawn from [pseed]. *)
type production = { heavy : bool; count : int; pseed : int }

let pp_production ppf c =
  Format.fprintf ppf "%d %s launches, seed %d" c.count
    (if c.heavy then "heavy" else "light")
    c.pseed

let production =
  let open Gen in
  with_pp pp_production
    (let* heavy = bool in
     let+ count =
       if heavy then int_range 300 400
       else int_range (Plan.part_launches + 1) (Plan.part_launches + 1000)
     and+ pseed = nat in
     { heavy; count; pseed })

let production_examples =
  [
    { heavy = true; count = 300; pseed = 1 };
    { heavy = false; count = Plan.part_launches + 1; pseed = 2 };
  ]

(* The most bytes a one-group launch of [kernel] costs within [s] seconds. *)
let bytes_within kernel s =
  let time bytes =
    Plan.seconds
      ~threads:(Nx_metal.Kernels.threads kernel)
      ~groups:(1, 1, 1) ~resident:1 (cost ~bytes ())
  in
  let rec go lo hi =
    if hi - lo <= 1 then lo
    else
      let m = lo + ((hi - lo) / 2) in
      if time m <= s then go m hi else go lo m
  in
  go 0 (1 lsl 50)

(* plan.mli: at part_budget and part_launches, the first part of a call of
   several holds at least 199 launches, more bytes than any one launch, so no
   later part passes it. Planned only. *)
let law_production c =
  let st = Random.State.make [| c.pseed |] in
  let b = Plan.make () in
  for _ = 1 to c.count do
    let kernel = part_kernels.(Random.State.int st 3) in
    let bytes =
      if not c.heavy then 4
      else
        bytes_within kernel (Plan.budget *. (0.8 +. Random.State.float st 0.2))
    in
    let slot = Plan.slot b (Random.State.int st 4) in
    Plan.launch b kernel ~groups:(1, 1, 1) ~cost:(cost ~bytes ())
      (Plan.words kernel)
      [| { Plan.word = 0; slot; write = true } |]
  done;
  let call = Plan.finish b in
  cover "cut by seconds" c.heavy;
  cover "cut by launches" (not c.heavy);
  greater ~msg:"parts" int ~than:1 (Array.length call.parts);
  let first = call.parts.(0).run in
  at_least ~msg:"the first part's launches" int ~than:199 (Array.length first);
  let fa, fv = run_bytes first in
  Array.iteri
    (fun i (p : Plan.part) ->
      let a, v = run_bytes p.run in
      let msg = Printf.sprintf "part %d" i in
      at_most ~msg:(msg ^ "'s parameters") int ~than:fa a;
      at_most ~msg:(msg ^ "'s values") int ~than:fv v;
      equal
        ~msg:(msg ^ " commits past commit_after")
        bool
        (run_time p.run > Plan.commit_after)
        p.commits)
    call.parts

(* A plan to weigh: [count] launches, each of a kernel and a count of refs drawn
   from [hseed], cut into parts of at most [part] launches. *)
type held = { count : int; part : int; hseed : int }

let pp_held ppf h =
  Format.fprintf ppf "%d launches, parts of %d, seed %d" h.count h.part h.hseed

let held =
  let open Gen in
  with_pp pp_held
    (let* count = int_range 1 2000 in
     let+ part = int_range 1 count and+ hseed = nat in
     { count; part; hseed })

let held_kernels =
  Nx_metal.Kernels.[| Zero; Copy_run; Copy; La_trsm_f32; La_qr_panel_f32 |]

(* plan.mli: a call holds at most [184 + 8 w + 40 r] bytes of host memory for
   each launch of [w] words and [r] refs, [32] more where its threads are not
   its kernel's, [128 + 16 s] for each part and [128 + 24 s] for the call, of
   [s] slots, beside its slots' own bytes. Each launch here takes its own words,
   refs, groups and cost, so none shares with another. What a call holds is the
   live heap it adds, its builder collected. *)
let held_plan h =
  let st = Random.State.make [| h.hseed |] in
  let b = Plan.make () in
  let bound = ref 0 in
  for _ = 1 to h.count do
    let kernel = held_kernels.(Random.State.int st 5) in
    let w = Nx_metal.Kernels.words kernel in
    let r = 1 + Random.State.int st Plan.most_refs in
    let most = Nx_metal.Kernels.threads kernel in
    let threads = if Random.State.bool st then most else Int.max 1 (most / 2) in
    let refs =
      Array.init r (fun i ->
          { Plan.word = i mod w; slot = Plan.slot b (i mod 4); write = i = 0 })
    in
    Plan.launch b kernel
      ~groups:(Sys.opaque_identity 1, 1, 1)
      ~threads ~cost:(cost ~bytes:4 ()) (Plan.words kernel) refs;
    bound :=
      !bound + 184 + (8 * w) + (40 * r) + if threads = most then 0 else 32
  done;
  (S.parted ~launches:h.part (fun () -> Plan.finish b), !bound)

let live () =
  Gc.full_major ();
  (Gc.quick_stat ()).live_words

let law_held h =
  let before = live () in
  let c, launches = held_plan h in
  let held = 8 * (live () - before) in
  let s = Array.length c.slots and parts = Array.length c.parts in
  cover "one part" (parts = 1);
  cover "several parts" (parts > 1);
  cover "a part a launch" (parts = h.count && parts > 1);
  at_most ~msg:"held bytes" int
    ~than:(launches + (parts * (128 + (16 * s))) + 128 + (24 * s))
    held

let held_examples =
  [
    { count = 1; part = 1; hseed = 0 };
    { count = 2000; part = 2000; hseed = 1 };
    { count = 2000; part = 1; hseed = 2 };
  ]

(* plan.mli: parts of the same kernels and refs share a key; a part keeps its
   submission by its own launches. *)
let test_part_keys () =
  let b = Plan.make () in
  let slot = Plan.slot b 0 in
  for _ = 1 to 6 do
    Plan.launch b Nx_metal.Kernels.Zero ~groups:(1, 1, 1) ~cost:(cost ())
      (Plan.words Nx_metal.Kernels.Zero)
      [| { Plan.word = 0; slot; write = true } |]
  done;
  let c = S.parted ~launches:2 (fun () -> Plan.finish ~most:2 b) in
  equal ~msg:"parts" int 3 (Array.length c.parts);
  let k = Plan.key c.parts.(0) in
  Array.iter
    (fun (p : Plan.part) ->
      equal ~msg:"one key" bool true (Plan.matches k p);
      equal ~msg:"kept by its own launches" bool true p.kept)
    c.parts

let test_part_bounds () =
  raises (Invalid_argument "Nx_metal.bound_parts: launches") (fun () ->
      Plan.bound_parts ~launches:0 ~seconds:1.);
  raises (Invalid_argument "Nx_metal.bound_parts: launches") (fun () ->
      Plan.bound_parts ~launches:(Plan.part_launches + 1) ~seconds:1.);
  raises (Invalid_argument "Nx_metal.bound_parts: seconds") (fun () ->
      Plan.bound_parts ~launches:1 ~seconds:0.);
  raises (Invalid_argument "Nx_metal.bound_parts: seconds") (fun () ->
      Plan.bound_parts ~launches:1 ~seconds:(Plan.part_budget *. 2.))

(* plan.mli: no result depends on the parts. A rotation's copies stage operands
   through the workspace, live across the launches a cut separates. *)
let law_rotation_parts (c, r) = parted c (fun () -> law_rotation r)

(* Swaps through the workspace, in launches of a microsecond's work: a long
   swap's copies in tens of launches, a short one's in one each. *)
let swap_elements = 16_384
let short_elements = 64
let swap_budget = Plan.dispatch +. 1e-6

let swap_bits n =
  let bits seed = random_bytes (Random.State.make [| seed |]) (4 * n) in
  (bits (n + 1), bits (n + 2))

let swap n =
  let a, b = swap_bits n in
  let draw s =
    let x = A.create (S.device ()) D.Int32 [| n |] in
    Rig.Buffer.blit_from_string s 0 (A.buffer x) 0 (4 * n);
    x
  in
  let bits x =
    let s = Bytes.create (4 * n) in
    Rig.Buffer.blit_to_bytes (A.buffer x) 0 s 0 (4 * n);
    Bytes.unsafe_to_string s
  in
  let x = draw a and y = draw b in
  let ins : D.any iarray = [| D.Any D.Int32; D.Any D.Int32 |] in
  let p = P.v ~ins [| P.In 0; P.In 1 |] ~outs:[| 1; 0 |] in
  let dsts : A.any iarray = [| A.Any x; A.Any y |] in
  let loads : Nx_kernel.load iarray = [| Plain x; Plain y |] in
  equal answer A.Done (Nx_metal.map p ~dsts loads);
  (bits x, bits y)

(* The launches of nx.metal's plan of a swap of [n] elements under
   [swap_budget], planned on the host's arrays alone. *)
let swap_launches n =
  let x = A.create Rig.host D.Int32 [| n |]
  and y = A.create Rig.host D.Int32 [| n |] in
  let ins : D.any iarray = [| D.Any D.Int32; D.Any D.Int32 |] in
  let p = P.v ~ins [| P.In 0; P.In 1 |] ~outs:[| 1; 0 |] in
  S.parted ~budget:swap_budget (fun () ->
      match moves_map p ~dsts:[| A.Any x; A.Any y |] [| Plain x; Plain y |] with
      | Ok (Some (c : Plan.t)) -> Array.length c.launches
      | Ok None | Error _ -> failwith "the moves planned no swap")

(* The launches of a part: a third of a long swap's, so that it runs as three
   parts, and at least a short swap's, so that the short swap is one part and
   uses the device's workspace. Planned as the program starts, before any test's
   domains run, since the bounds are global. *)
let part_launches =
  Int.max (swap_launches short_elements) ((swap_launches swap_elements + 2) / 3)

let swapped n () =
  S.parted ~budget:swap_budget ~launches:part_launches (fun () -> swap n)

(* plan.mli: a call of several parts takes a workspace of its own, which no
   other call writes between two of its parts: a long swap in three parts and
   short swaps of the device's workspace, on two domains, each get the other's
   bits. *)
let domains_parts_workspace =
  let want n () =
    let a, b = swap_bits n in
    (b, a)
  in
  let swap = Gen.unit @-> returns (pair string string) in
  stateful ~domains:2 ~count:2 "a call's parts and other calls get their bits"
    [
      command "long" swap (want swap_elements) (swapped swap_elements);
      command "short" swap (want short_elements) (swapped short_elements);
    ]

(* nx_metal.mli: a call of several parts computes alike on its first run, with
   its parts kept and with none kept, and its parts each one part's launches. *)
let test_parts_kept () =
  let a, b = swap_bits swap_elements in
  let _, launches = Plan.traced (swapped swap_elements) in
  equal ~msg:"launches, as planned on the host" int
    (swap_launches swap_elements)
    (List.length launches);
  greater ~msg:"launches" int ~than:part_launches (List.length launches);
  let run () = swapped swap_elements () in
  equal ~msg:"first" (pair string string) (b, a) (run ());
  equal ~msg:"kept" (pair string string) (b, a) (run ());
  Fun.protect
    ~finally:(fun () -> Nx_metal.bound_kept ~launches:max_int)
    (fun () ->
      Nx_metal.bound_kept ~launches:0;
      equal ~msg:"none kept" (pair string string) (b, a) (run ()))

let () =
  exit
    (run "nx_metal"
       [
         group "copies"
           [
             prop "a copy keeps the bits around its destination" around
               law_around;
             prop ~examples:rotation_examples
               "outputs that read each other's destinations get the operands \
                the call found"
               rotation law_rotation;
             (* At 400 cases, every run draws ends on 1 byte, the rarest case,
                at about 5 %. *)
             prop ~count:400 ~examples:source_examples
               "a copy of a source of any layout gives its elements and keeps \
                the bits around its destination"
               source law_source;
             prop ~count:200 ~examples:wide_examples
               "a copy of 64-bit indices, under bound_copy, gives the elements \
                of 32-bit ones"
               source_wide law_source_wide;
             prop ~count:300 ~examples:rows_examples
               "a copy of rows of 16 bytes or more runs copy_rows and gives \
                its elements"
               rows_source law_rows;
             prop ~count:200 ~examples:rows_examples
               "a copy of rows of 64-bit byte indices, under bound_copy, gives \
                its elements"
               rows_wide law_rows_wide;
             prop ~count:300 ~examples:steps_examples
               "a copy of every other element in rows of its width's least \
                bytes runs copy_step and gives its elements"
               steps_source law_rows;
             prop ~count:100 ~examples:steps_examples
               "a copy of every other element past bound_copy's bound copies \
                an element a thread and gives its elements"
               steps_wide law_rows_wide;
             test "an operand another output writes is staged in workspace"
               test_staged;
             test "a broadcast operand stages the elements it reads"
               test_staged_broadcast;
             test "swaps hold one call's workspace of their own" test_fresh_held;
             test "a call takes the device's tickets or tickets of its own"
               test_tickets_door;
             domains_copy_alike;
           ];
         group "kept"
           [
             test "a copy holds 8 axes in its parameters, more in a constant"
               test_holds;
             test
               "a device keeps 16 constants, and a call its own past the bound"
               test_constants;
             test "a call past the launches kept keeps no submission"
               test_unkept;
             test "bounds out of range are refused" test_bounds;
             test "keys of one shape twice fall in buckets as uniform draws do"
               test_hash_spread;
             test
               "a kept call at another base reads and writes there, by the \
                direct kernel and by the tape"
               test_kept_bases;
             prop ~examples:planned_examples
               "a map traces and computes alike where it plans, where it finds \
                its plan and with none kept"
               planned_map law_planned;
             prop ~examples:facts_examples
               "calls of one program whose facts differ each plan their own"
               facts_case law_facts;
             test "adds at two lengths share a submission" test_shared_kernel;
             domains_share_runs;
           ];
         group "parts"
           [
             prop ~examples:cuts_examples
               "a call's launches are cut into parts within their bounds"
               planned_parts law_cuts;
             prop ~count:6 ~examples:production_examples
               "at the library's bounds no later part passes the first's bytes"
               production law_production;
             prop ~count:20 ~examples:held_examples
               "a plan holds its stated host bytes a launch" held law_held;
             test "parts of the same launches share a key" test_part_keys;
             test "part bounds out of range are refused" test_part_bounds;
             prop
               ~examples:(List.map (fun r -> (Launches 2, r)) rotation_examples)
               "a rotation cut into parts keeps its bits"
               (Gen.pair cut rotation) law_rotation_parts;
             test "a call of several parts computes alike, kept or not"
               test_parts_kept;
             test "parts of one key share one kept submission" test_parts_share;
             domains_parts_workspace;
             prop ~count:20 ~examples:split_scan_examples
               "a scan split across launches, a part each, has nx.cpu's bits"
               split_scan (fun s ->
                 S.parted ~launches:1 (fun () -> law_split_scan s));
             prop ~count:20 ~examples:windowed_examples
               "a contraction cut into windows, chunks and parts keeps its bits"
               windowed_case (fun w ->
                 S.parted ~launches:1 (fun () -> law_window_bits w));
           ];
         group "budget"
           [
             test "a launch past the budget is refused" test_refused;
             test "a launch takes at most 13 refs" test_most_refs;
             prop ~examples:narrow_examples
               "an estimate is at least the time its counts take" narrow
               law_narrow;
             prop "each launch keeps within the time budget"
               ~examples:budget_examples budgeted law_budget;
             prop
               "a copy of 2^32 elements or more keeps each launch within the \
                time budget"
               huge law_huge;
             test "a strided copy counts a line a load" test_strided_reads;
             test "copy_step weighs the elements it copies one at a time"
               test_step_serial;
           ];
         group "harness"
           [
             prop ~examples:tickets_examples
               "the last group to take a ticket sees every part" tickets
               law_tickets;
           ];
         group "loops"
           [
             prop ~count:300
               ~examples:(direct_examples @ direct_rare)
               "a map of a direct form has the tape's bits, and runs the tape \
                where bound"
               direct law_direct;
             test
               "a narrow float's direct map of operands 2 or 6 bytes into \
                their buffers has the tape's bits"
               test_direct16_shifted;
             test
               "a direct map of bytes runs the kernel of bytes, of a long kind \
                the direct kernel, with the tape's bits"
               test_direct8_kinds;
             test "a direct map of 8 axes runs the direct kernel, of 9 the tape"
               test_direct_rank;
             test
               "a map that leaves a load unread runs the direct kernel, with \
                nx.cpu's bits"
               test_unread_load;
             test
               "a transposed cast takes the tape's tiles where their estimate \
                is the lower"
               test_cast_transposed;
             test "every float16 code decodes in its cast's kernel"
               test_f16_codes;
             test "the tape has nx.cpu's float16 bits at every code"
               test_f16_tape;
             cases ~id:cast_id
               "a cast has nx.cpu's bits, by its class's kernel, the copies or \
                the tape"
               cast_cases law_cast;
             test "a cast runs its tape where bound, with the same bits"
               test_cast_bound;
             test
               "a cast of uint4 into uint8 keeps each value past 2^30 elements"
               test_nibble_cast_far;
             test
               "a pass of a map past a launch's buffers whose outputs move \
                bits copies them"
               test_split_moves;
             test
               "a move into a stepped or reversed destination, alone or in a \
                split map, stores its operand's bits and nothing around them"
               test_moves_into_views;
             test "a split map's copies count in its parts' seconds"
               test_split_budget;
             prop ~count:300 ~examples:reduced_examples
               "a reduction of one load by Sum, Max, Min, Arg, Logsumexp or \
                Moments has the tape's bits, and runs the tape where bound"
               reduced law_reduce_direct;
             prop ~count:300 ~examples:one_term_examples
               "an output of one term is its term folded from its monoid's \
                start, with nx.cpu's bits"
               one_term law_one_term;
             prop ~count:300
               "a scan's result of one term is its term folded from its carry \
                or its monoid's start, with nx.cpu's bits"
               one_scan law_one_term_scan;
             prop ~count:300 ~examples:row_examples
               "a reduction of a float load along a row runs its reduction's \
                and dtype's kernel, with the tape's bits"
               (rowed ~along:true) law_row_direct;
             prop ~count:1000
               "a direct reduction takes a row's kernel exactly where its load \
                is a float's and its reduced axes one axis"
               (rowed ~along:false) law_row_rule;
             prop "kinds flush as nx_metal.mli states" (named flush_dtypes)
               test_flush;
             prop
               "kinds and reductions that order values are exact about zero \
                and at every NaN"
               (named order_dtypes) test_order;
             prop ~count:100
               ~examples:
                 (* Merges by a launch and by a ticket, of int32 sums by
                    loop_reduce_direct and float32 ones by a row's. *)
                 (List.concat_map
                    (fun int32 ->
                      List.map
                        (fun apart ->
                          {
                            groups = 4;
                            outputs = 1;
                            rounds = 1;
                            apart;
                            direct = true;
                            int32;
                          })
                        [ true; false ])
                    [ true; false ])
               "a sum's last group sees every group's state, from 1 to 128 \
                groups an output"
               summed law_summed;
             test "a launch counts a core's groups by its pipeline's threads"
               test_resident;
             test "a reduction of no term declines as one of a term does"
               test_no_term;
             prop ~count:100 ~examples:short_rows_examples
               "a reduction of 1 to 16 terms an output has nx.cpu's bits"
               short_rows law_short_rows;
             prop ~count:100 ~examples:short_window_examples
               "a reduction of windows of few terms has nx.cpu's bits"
               short_windows law_short_windows;
             prop ~count:40
               ~examples:
                 [
                   (* Seed s1:a3fb46ff985922cd: a -inf among terms whose product
                      is subnormal on the CPU, -inf there, and zero on the GPU,
                      NaN there. *)
                   {
                     pc = class_in short_cases "a bfloat16 product";
                     pshape = [| 1; 174; 1 |];
                     place = 0;
                     pseed = 234;
                   };
                   (* A sum, maxima and an Arg of 5 blocks, at budgets where a
                      thread takes 4 elements and each group's runs take several
                      launches: by a row's sum, loop_reduce_direct, a row's
                      maximum and a row's Arg. *)
                   {
                     pc = class_in short_cases "a float32 sum";
                     pshape = [| 1; 5000; 1 |];
                     place = 9;
                     pseed = 1;
                   };
                   {
                     pc = class_in short_cases "an int32 maximum";
                     pshape = [| 1; 5000; 1 |];
                     place = 5;
                     pseed = 1;
                   };
                   {
                     pc = class_in short_cases "a float32 maximum";
                     pshape = [| 1; 5000; 1 |];
                     place = 9;
                     pseed = 1;
                   };
                   {
                     pc = class_in short_cases "an Arg of float32";
                     pshape = [| 1; 5000; 1 |];
                     place = 9;
                     pseed = 1;
                   };
                   (* A Logsumexp, an output of one block and one of several
                      groups. *)
                   {
                     pc = class_in short_cases "a float32 Logsumexp";
                     pshape = [| 1; 5000; 1 |];
                     place = 0;
                     pseed = 1;
                   };
                   {
                     pc = class_in short_cases "a float32 sum";
                     pshape = [| 2; 500; 1 |];
                     place = 0;
                     pseed = 1;
                   };
                   {
                     pc = class_in short_cases "a float32 sum";
                     pshape = [| 1; 20000; 1 |];
                     place = 0;
                     pseed = 1;
                   };
                 ]
               "a reduction's group in several launches has nx.cpu's bits but \
                where a combine underflows"
               pieces law_pieces;
             prop ~count:40 ~examples:split_scan_examples
               "a scan split across launches has nx.cpu's bits" split_scan
               law_split_scan;
             prop ~count:80
               ~examples:
                 (* 16 slices of 5 tiles of 16 terms: tile 3's end joins the
                    network of tile blocks 0 to 3, which a running fold of the
                    tiles' ends does not give. *)
                 (List.map
                    (fun nc ->
                      {
                        nc;
                        nshape = [| 1; 80; 16 |];
                        nbudget = Plan.budget;
                        nseed = 1;
                      })
                    [ 0; 1 ]
                 (* 256 slices of 1 term and 128 of 2, a group each, and a slice
                    of 5 blocks, whose block 3 ends at the chain's result. *)
                 @ [
                     {
                       nc = 0;
                       nshape = [| 1; (5 * 4096) + 1; 1 |];
                       nbudget = Plan.budget;
                       nseed = 5;
                     };
                     {
                       nc = 0;
                       nshape = [| 1; 1; 256 |];
                       nbudget = Plan.budget;
                       nseed = 2;
                     };
                     {
                       nc = 1;
                       nshape = [| 1; 2; 128 |];
                       nbudget = Plan.budget;
                       nseed = 3;
                     };
                     (* 32 slices a group, which 80 draws can miss, and slices
                        of 7 blocks and a half in launches of 0.1 ms, their
                        blocks' chain in several. *)
                     {
                       nc = 0;
                       nshape = [| 2; 40; 16 |];
                       nbudget = Plan.budget;
                       nseed = 4;
                     };
                     {
                       nc = 1;
                       nshape = [| 2; (7 * 4096) + 2048; 16 |];
                       nbudget = 1e-4;
                       nseed = 6;
                     };
                   ])
               "a Logsumexp or Moments scan has nx.cpu's bits by its tiles"
               net_scan law_network_scan;
             test "a 1-D logcumsumexp of 2^21 terms has nx.cpu's bits"
               test_long_network;
             test
               "a network of several blocks a slice takes the workspace \
                nx_metal.mli states"
               test_network_workspace;
             test "a long scan is planned at every rank to 32" test_kept_scans;
             test
               "a map whose run of 4 elements a thread passes a launch is \
                planned"
               test_costly_maps;
             test
               "a Logsumexp or Moments scan of 2^28 elements is planned at \
                every length a slice"
               test_network_lengths;
             test "long scans take the time nx_metal.mli estimates"
               test_estimates;
             test
               "direct reductions along a row take their launches, their \
                operands' bytes counted as nx_metal.mli says"
               test_row_estimates;
             test
               "reductions of outputs side by side and of few terms are \
                charged the lines their groups touch"
               test_box_estimates;
             test "a loop's launches count every line their operands touch"
               test_short_runs;
             test
               "a transposed read counts a line an element, its touches \
                farther apart than the cache"
               test_far_transpose;
             test "a staged transposed read counts at most twice its bytes"
               test_staged_transpose;
             test
               "a broadcast row counts its lines a row, its touches farther \
                apart than the cache"
               test_far_broadcast;
             prop ~count:200 ~examples:staged_examples
               "a map whose loads are staged has nx.cpu's bits" staged_cases
               law_staged;
             test
               "a transposed load is staged where the rule says, and only there"
               test_staged_plans;
             test "a map takes tiles where its stages fit and gain"
               test_tile_choice;
             test "a part that reads a transpose takes tiles" test_staged_part;
             test "70 values live at once run in parts with nx.cpu's bits"
               test_wide_values;
             test "tiles past a plane's last, in later launches"
               test_staged_launches;
             prop ~count:200 ~examples:padded_examples
               "a map of padded loads has nx.cpu's bits" padded_cases law_padded;
             prop ~count:200 ~examples:axis_examples
               "a map along one axis has nx.cpu's bits at any length" axis_map
               law_axis_map;
             test "a padded coordinate at 2^31 takes 64 bits" test_padded_reach;
             test "a header names its map's kernel, or no kernel raises"
               test_kernel_of;
             test "t_add takes the time nx_metal.mli estimates"
               test_tile_estimates;
             prop ~count:300 ~examples:lines_examples
               "a loop's launches count at least the lines its operands touch, \
                at any layout"
               lines_cases law_lines;
             prop
               ~examples:
                 (* A scan from a carry, two classes of 26, which 100 draws can
                    miss. *)
                 [
                   {
                     kc = long_class "a float32 sum from a carry";
                     kshape = [| 5000; 2 |];
                     kstep = false;
                     kholds = 256;
                     near = false;
                   };
                 ]
               "a drawn long scan is planned" kept_scan law_kept_scan;
             prop ~count:20
               ~examples:
                 (* The classes of the most nodes of each kind, a block the
                    budget cuts below 16 tiles, and launches that hold less than
                    a wave. *)
                 [
                   {
                     gc = long_class "an Arg of complex64";
                     grank = 32;
                     gslices = 14;
                     gholds = 3072;
                     gstep = true;
                   };
                   {
                     gc = long_class "Moments of a bfloat16 Fma";
                     grank = 8;
                     gslices = 9;
                     gholds = 256;
                     gstep = true;
                   };
                   {
                     gc = long_class "a sum of 2300 float32 Exps";
                     grank = 8;
                     gslices = 9;
                     gholds = 256;
                     gstep = false;
                   };
                   {
                     gc = long_class "Moments of 2300 float32 Exps";
                     grank = 8;
                     gslices = 9;
                     gholds = 256;
                     gstep = false;
                   };
                   {
                     gc = long_class "a Logsumexp of 160 float32 Exps";
                     grank = 8;
                     gslices = 9;
                     gholds = 256;
                     gstep = false;
                   };
                   {
                     gc = long_class "an Arg of 1800 complex64 Pows";
                     grank = 8;
                     gslices = 9;
                     gholds = 256;
                     gstep = false;
                   };
                   {
                     gc = long_class "a Logsumexp of 2300 float32 Exps";
                     grank = 8;
                     gslices = 9;
                     gholds = 256;
                     gstep = true;
                   };
                 ]
               "a scan plans at every length, past a part's seconds in parts"
               grown law_grown_plans;
             prop ~examples:looped_examples
               "each launch of a loop keeps within the time budget" looped
               law_loop_budget;
           ];
         group "contraction plans"
           [
             prop ~examples:plan_examples
               "a plan's tiles and ranges follow the rule from the extents"
               contracted law_plan;
             prop ~examples:window_examples
               "a plan cuts its tiles into windows of whole units within any \
                budget"
               contracted law_windows;
             cases
               ~id:(fun (name, _, _) -> name)
               "the design's calls take the launches it states" launch_rows
               test_launches;
             prop "a Rows call's plan follows the rule from the extents"
               grouped_call law_grouped_plan;
             cases
               ~id:(fun (k, _) -> Printf.sprintf "%d terms" k)
               "a vector plan deals an output's spans to its lanes"
               vector_lane_rows test_vector_lanes;
           ];
         group "contraction bits"
           [
             prop ~count:40 ~examples:bits_examples
               "a contraction has the bits of its stated association" bits_case
               law_bits;
             prop ~count:20 ~examples:windowed_examples
               "a contraction cut into windows and chunks keeps its bits"
               windowed_case law_window_bits;
             prop ~count:30 ~examples:cbits_examples
               "a complex64 contraction has its real form's bits" cbits_case
               law_complex_bits;
             prop ~count:30 ~examples:fheld_examples
               "a float32 contraction takes init and dst of any dtype by their \
                casts"
               fheld_case law_float_held;
             prop ~count:40 ~examples:cvbits_examples
               "a float32 contraction takes operands of any dtype by their \
                casts"
               cvbits_case law_converted_bits;
             prop ~count:30 ~examples:scbits_examples
               "a scaled float32 contraction takes each element times its scale"
               scbits_case law_scaled_bits;
             prop ~count:30 ~examples:mixed_examples
               "a float32 contraction of operands not of its stage's dtype is \
                the call of their casts"
               scbits_case law_mixed_casts;
             prop ~count:20 ~examples:mixed_pieces_examples
               "a float32 contraction whose casts pass its buffer is cut into \
                pieces of its bits"
               mixed_pieces_case law_mixed_pieces;
             test "a cast line of several runs past the buffer is cast whole"
               (band_runs ~cast_b:false);
             test
               "a line cast whole leaves the other cast line a band of what \
                remains"
               (band_runs ~cast_b:true);
             prop ~count:200 ~examples:pcase_examples
               "a contraction's pieces partition its outputs within the buffer"
               pcase law_cut_pieces;
             prop ~count:200
               "a refined piece partitions its box within the buffer"
               Gen.(pair pcase (int_range 0 40))
               law_refine_pieces;
             prop ~count:40 ~examples:dst_pieces_examples
               "a contraction into a dst apart or of more outputs than a \
                piece's computes in pieces of its bits"
               dcase law_dst_pieces;
             cases ~id:scale_row_id
               "a scale multiplies an element as its kind's rule says"
               scale_rows test_scale_row;
             cases ~id:step_row_id "a loader's scale steps lie within 32 bits"
               step_rows test_step_row;
             prop ~examples:division_examples
               "a loader's divisor word divides every 32-bit term" division
               law_divisor;
             test "a divisor word is of 1 to 2^32 - 1" test_divisor_domain;
             cases ~id:weight_id
               "a weight as a model stores it takes its key's loader" weights
               test_weight;
             test "an MXFP4 loader scales each element as its code says"
               test_exponent_edges;
             cases ~id:weight_name
               "experts of a weight as a model stores it take loaders" experts
               test_experts;
             cases ~id:weight_name
               "experts of a weight as a model stores it take the vector \
                kernels at few rows"
               experts test_vector_experts;
             prop ~count:30 ~examples:blocks_examples
               "a weight of blocks of terms takes loaders where it lies" bcase
               law_blocks;
             prop ~count:40 ~examples:vcall_examples
               "a call of few rows over a key's weight has the vector kernels' \
                bits"
               vcall_case law_vector_bits;
             cases ~id:weight_id
               "a weight as a model stores it takes the vector kernels at few \
                rows"
               vector_weights test_vector_weight;
             cases ~id:vector_rows_id
               "a vector call at its key's bound of terms finds launches"
               vector_rows_cases test_vector_terms;
             cases ~id:terms_id
               "a call past the vector kernels' terms plans on the tiles"
               terms_cases test_terms;
             cases ~id:vector_terms_id
               "a call one term past its key's bound plans on the tiles"
               vector_terms_cases test_tiles_past_terms;
             cases ~id:dense_id
               "a dense weight of columns contiguous takes the vector kernels \
                where it lies"
               vector_dense test_vector_dense;
             prop ~count:40 ~examples:ibits_examples
               "an integer contraction is exact modulo its accumulator"
               ibits_case law_int_bits;
             prop ~count:30 ~examples:iwindowed_examples
               "an integer contraction cut into windows and chunks keeps its \
                bits"
               iwindowed_case law_int_window_bits;
             prop ~count:0 ~examples:imax_rows
               "an integer contraction at its operands' maxima is exact"
               (Gen.with_pp pp_imax (Gen.of_list imax_rows))
               law_int_maxima;
             prop ~count:40 ~examples:rbits_examples
               "a Rows contraction multiplies each row by its group's b"
               rbits_case law_rows_bits;
             prop ~count:30 ~examples:rconverted_examples
               "a Rows contraction takes operands of any dtype by their casts"
               rconverted_case law_rows_converted;
             prop ~count:20 ~examples:rscaled_examples
               "a Rows contraction takes scaled operands by their scaled values"
               rscaled_case law_rows_converted;
             prop ~count:15 ~examples:rows_pieces_examples
               "a Rows contraction whose casts pass its buffer is cut into \
                pieces of its bits"
               rows_pieces_case law_rows_pieces;
             prop ~count:20 ~examples:rwindowed_examples
               "a Rows contraction cut into windows and chunks keeps its bits"
               rwindowed_case law_rows_window_bits;
             prop ~count:40 ~examples:kbits_examples
               "a Contracting contraction sums each group's terms alone"
               kbits_case law_kbits;
             prop ~count:20 ~examples:kscaled_examples
               "a Contracting contraction takes scaled operands by their \
                scaled values"
               kscaled_case law_kscaled;
             prop ~count:20 ~examples:kwindowed_examples
               "a Contracting contraction cut into windows and chunks keeps \
                its bits"
               kwindowed_case law_k_window_bits;
           ];
         group "contraction declines"
           [
             cases ~id:fst "a contraction declines what it does not compute"
               decline_cases test_declines;
             test "a contraction past 10 s runs in parts, one within in one"
               test_part_budget;
           ];
       ])
