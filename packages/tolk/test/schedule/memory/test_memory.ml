(* Tests of Tolk.Memory: the plans of recorded and generated schedules, and the
   buffers and storage an entry of a schedule takes. *)

open Windtrap
open Tolk

let uop = Uops.uop
let cpu = "CPU"

let buf ?(device = cpu) ?(dt = Dtype.Int8) ?phase size =
  Ops.new_buffer ?phase device size dt

let call args = Ops.call (Ops.sink args) args
let linear entries = Ops.v Linear ~src:entries

let loop entries =
  Ops.end_ (linear entries)
    [ Ops.range ~axis_type:Loop (Int 4) [ Ops.unique_num () ] ]

(* The storage [u] makes that [linear] does not hold, the arenas. *)
let arenas linear u =
  let held = Ops.backward_slice_with_self ~calls:Skip linear in
  List.filter
    (fun n -> Ops.op n = Buffer && not (Ops.Nodes.mem n held))
    (Ops.toposort ~calls:Enter u)

(* The sizes of the arenas of [linear]'s plan. *)
let arena_bytes ?held_bufs linear =
  List.map Shape.max_numel
    (arenas linear (Memory.memory_plan_rewrite ?held_bufs linear))

(* Recorded schedules *)

(* [numbered_as golden linear u] is [u] with its arenas numbered as [golden]'s:
   new storage takes its number from a counter the process shares, so the
   numbers are no property of the plan. *)
let numbered_as golden linear u =
  let slot n =
    match Ops.arg n with Param p -> p | _ -> fail "an arena has a parameter"
  in
  Ops.substitute ~calls:Skip ~pass:Fixed_point u
    (List.map2
       (fun mine theirs ->
         ( mine,
           Ops.replace
             ~arg:(Param { (slot mine) with slot = (slot theirs).slot })
             mine ))
       (arenas linear u) (arenas linear golden))

let schedule name = Golden.sink (name ^ ".golden")
let held name = Ops.src (Golden.sink (name ^ "_held.golden"))

let plan name =
  Memory.memory_plan_rewrite ~held_bufs:(held name) (schedule name)

let recorded_schedules =
  [
    "simple";
    "some_held";
    "all_held";
    "reused";
    "very_small";
    "big";
    "copy_apart_from_compute";
    "copies_share";
    "computes_share";
    "copies_held_mixed";
    "copy_chain";
    "sizes";
    "disk";
    "random_0";
    "random_1";
    "random_2";
    "random_3";
    "random_4";
    "random_5";
    "random_6";
    "random_7";
    "random_8";
    "random_9";
    "random_10";
    "random_11";
    "softmax";
    "attention";
    "matmul_chain";
    "convs";
    "copies";
    "sort";
  ]

let recorded =
  group "plans each recorded schedule as its golden states"
    (List.map
       (fun name ->
         let file = name ^ "_planned.golden" in
         Golden.graph file (fun () ->
             numbered_as (Golden.sink file) (schedule name) (plan name)))
       recorded_schedules)

(* Placed largest first, no recorded schedule's arenas are larger than those of
   tinygrad's allocator (DIVERGENCES.md D477, its reason). A rewrite of a
   [.tolk] by [-u] that keeps a larger plan fails here. *)
let no_larger =
  cases "a recorded plan's arenas take no more bytes than tinygrad's" ~id:Fun.id
    recorded_schedules (fun name ->
      let linear = schedule name in
      let bytes u =
        List.fold_left (fun n a -> n + Shape.max_numel a) 0 (arenas linear u)
      in
      let upstream = Golden.upstream (name ^ "_planned.golden") in
      at_most int ~than:(bytes upstream) (bytes (plan name)))

let chomp s =
  if String.ends_with ~suffix:"\n" s then String.sub s 0 (String.length s - 1)
  else s

let printouts =
  group "with debug at 1, prints the memory saved as its golden states"
    (List.map
       (fun name ->
         Golden.text (name ^ "_debug.golden") (fun () ->
             Setting.context
               [ Setting.B (Setting.debug, 1) ]
               (fun () -> ignore (plan name));
             chomp (output ())))
       [ "simple"; "sizes"; "copies" ])

(* Generated schedules

   Schedules of calls of buffers on one or two devices, some calls copies
   between two, some arguments views of a buffer, some calls in loops, some
   buffers held. *)

(* How a call's argument reaches a buffer: as itself, or through a view. *)
type reach = Pass | Shrink | Bitcast | After

type drawn = {
  sizes : (int * Dtype.t * int * Ops.device) list;
      (* Each buffer's elements, type, phase and device. *)
  calls : ((int * reach) list * bool) list;
      (* Each call's buffers, each passed or viewed, and whether it copies its
         second buffer into its first. *)
  depths : int list;
      (* Each call's loop depth: consecutive calls of depth 1 or more run in one
         loop, and among them consecutive calls of depth 2 in a loop inside
         it. *)
  held : int list;
}

let pp_drawn ppf d =
  let pp_size (n, dt, p, device) =
    Format.asprintf "%d %a at %d%s" n Dtype.pp dt p
      (if device = cpu then "" else " on " ^ device)
  in
  let pp_arg (b, reach) =
    match reach with
    | Pass -> string_of_int b
    | Shrink -> Printf.sprintf "shrink %d" b
    | Bitcast -> Printf.sprintf "bitcast %d" b
    | After -> Printf.sprintf "after %d" b
  in
  let pp_call ppf (bs, copy) =
    Format.fprintf ppf "%s(%s)"
      (if copy then "copy" else "call")
      (String.concat ", " (List.map pp_arg bs))
  in
  Format.fprintf ppf "sizes [%s], %a, depths [%s], held [%s]"
    (String.concat "; " (List.map pp_size d.sizes))
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       pp_call)
    d.calls
    (String.concat "; " (List.map string_of_int d.depths))
    (String.concat "; " (List.map string_of_int d.held))

let drawn =
  let open Gen in
  let* n = int_range 1 10 in
  let buffer = int_range 0 (n - 1) in
  let+ sizes =
    list ~size:(constant n)
      (quad
         (of_list [ 1; 16; 255; 256; 257; 1000; 4096; 70_000 ])
         (of_list [ Dtype.Int8; Float32; Float16 ])
         (of_list [ 0; 0; 4; 12 ])
         (of_list [ cpu; cpu; cpu; "CPU:1" ]))
  and+ calls, depths =
    let* calls =
      list ~size:(int_range 1 12)
        (let* bs =
           list ~size:(int_range 1 4)
             (pair buffer
                (of_list [ Pass; Pass; Pass; Shrink; Bitcast; After ]))
         in
         let+ copy = bool in
         let copy =
           copy && List.length bs = 2 && fst (List.hd bs) <> fst (List.nth bs 1)
         in
         (bs, copy))
    in
    let+ depths =
      list ~size:(constant (List.length calls)) (of_list [ 0; 0; 1; 1; 2 ])
    in
    (calls, depths)
  and+ held = list ~size:(int_range 0 3) buffer in
  { sizes; calls; depths; held }

let drawn = Gen.with_pp pp_drawn drawn

let same_size : Dtype.t -> Dtype.t = function
  | Int8 -> Uint8
  | Float16 -> Int16
  | Float32 -> Int32
  | dt -> invalid_arg (Format.asprintf "no other type drawn for %a" Dtype.pp dt)

let build d =
  let buffers =
    Array.of_list
      (List.map
         (fun (n, dt, phase, device) -> buf ~device ~dt ~phase n)
         d.sizes)
  in
  (* A call passes a buffer, views its first element, views it as another type
     of one size, or names its state after a store. *)
  let arg (i, reach) =
    let b = buffers.(i) in
    match reach with
    | Pass -> b
    | Shrink -> Shape.shrink b [ Some (Int 0, Int 1) ]
    | Bitcast -> Ops.bitcast b (same_size (Ops.dtype b))
    | After -> Ops.after b [ Ops.store b b ]
  in
  let calls =
    List.map
      (fun (bs, copy) ->
        if copy then
          let dst, src = (fst (List.hd bs), fst (List.nth bs 1)) in
          Shape.store_call buffers.(dst) buffers.(src)
        else call (List.map arg bs))
      d.calls
  in
  (* Consecutive calls at depth [k] or more run in one loop of trips around
     those at depth [k + 1] or more. *)
  let loop entries =
    Ops.end_ (linear entries)
      [ Ops.range ~axis_type:Loop (Int 3) [ Ops.unique_num () ] ]
  in
  let rec group k = function
    | [] -> []
    | (_, depth) :: _ as calls when depth > k ->
        let inside, rest =
          let rec span acc = function
            | (c, dp) :: rest when dp > k -> span ((c, dp) :: acc) rest
            | rest -> (List.rev acc, rest)
          in
          span [] calls
        in
        loop (group (k + 1) inside) :: group k rest
    | (c, _) :: rest -> c :: group k rest
  in
  let entries = group 0 (List.combine calls d.depths) in
  (buffers, linear entries, List.map (fun i -> buffers.(i)) d.held)

(* The calls of [linear], those of its loops in their places. *)
let rec calls_of linear =
  List.concat_map
    (fun e -> if Ops.op e = End then calls_of (Ops.nth e 0) else [ e ])
    (Ops.src linear)

(* Each buffer's life, in calls, as [(first, last + 1)]: from the first call
   that takes it to the last, where every call that takes it lies in one loop;
   otherwise each use inside a loop that not all its uses lie in holds it across
   that loop's calls, the outermost such loop. *)
let lifetimes d =
  let depths = Array.of_list d.depths in
  let n = Array.length depths in
  (* The calls of the loop at depth [k] around call [i]: its first and last. *)
  let span i k =
    let lo = ref i and hi = ref i in
    while !lo > 0 && depths.(!lo - 1) >= k do
      decr lo
    done;
    while !hi < n - 1 && depths.(!hi + 1) >= k do
      incr hi
    done;
    (!lo, !hi)
  in
  let uses = Hashtbl.create 8 in
  List.iteri
    (fun i (bs, _) ->
      List.iter
        (fun (b, _) ->
          Hashtbl.replace uses b
            (i :: Option.value ~default:[] (Hashtbl.find_opt uses b)))
        bs)
    d.calls;
  fun b ->
    let us = Hashtbl.find uses b in
    (* The deepest loop every use lies in. *)
    let common =
      let rec deepest k =
        if k > 2 then 2
        else
          let spans = List.map (fun i -> span i k) us in
          if
            List.for_all (fun i -> depths.(i) >= k) us
            && List.for_all (fun s -> s = List.hd spans) spans
          then deepest (k + 1)
          else k - 1
      in
      deepest 1
    in
    let lo, hi =
      List.fold_left
        (fun (lo, hi) i ->
          let a, b =
            if depths.(i) > common then span i (common + 1) else (i, i)
          in
          (min lo a, max hi b))
        (max_int, min_int) us
    in
    (lo, hi + 1)

let meet (s0, e0) (s1, e1) = s0 < e1 && s1 < e0

(* The buffers the calls take that are not held, in the order they first
   appear. *)
let planned d =
  List.fold_left
    (fun seen (bs, _) ->
      List.fold_left
        (fun seen (b, _) ->
          if List.mem b seen || List.mem b d.held then seen else seen @ [ b ])
        seen bs)
    [] d.calls

let device d b =
  let _, _, _, device = List.nth d.sizes b in
  device

let phase d b =
  let _, _, phase, _ = List.nth d.sizes b in
  phase

(* A buffer's bytes of its arena: its size and phase, rounded up to 256. *)
let rounded d b =
  let n, dt, phase, _ = List.nth d.sizes b in
  (phase + (n * Dtype.itemsize dt) + 255) / 256 * 256

(* The placement the rule states: on each device, the planned buffers largest
   first, ties in the order they first appear, each at the start of the smallest
   gap that holds it between the buffers already placed whose lives meet its,
   else past the last byte of those. A gap is a run of bytes below that last
   byte that none of them takes. Of gaps of one size, the lowest. Each place
   comes with the number of gaps that held its buffer. *)
let reference d =
  let life = lifetimes d in
  let place placed b =
    let n = rounded d b in
    let spans =
      List.sort compare
        (List.filter_map
           (fun (b', o) ->
             if meet (life b) (life b') then Some (o, o + rounded d b')
             else None)
           (List.map (fun (b', o, _) -> (b', o)) placed))
    in
    let gaps, top =
      List.fold_left
        (fun (gaps, top) (o, e) ->
          ((if o > top then (o - top, top) :: gaps else gaps), max top e))
        ([], 0) spans
    in
    let holding = List.filter (fun (g, _) -> g >= n) gaps in
    let o =
      match List.sort compare holding with (_, o) :: _ -> o | [] -> top
    in
    placed @ [ (b, o, List.length holding) ]
  in
  let bufs = planned d in
  let devices = List.sort_uniq compare (List.map (device d) bufs) in
  List.concat_map
    (fun dev ->
      let mine = List.filter (fun b -> device d b = dev) bufs in
      List.fold_left place []
        (List.stable_sort
           (fun a b -> Int.compare (rounded d b) (rounded d a))
           mine))
    devices

(* [view arenas p] is the arena that [p], the replacement of a planned buffer,
   views, and the offset and length of the bytes it views. A shrink of the whole
   arena is the arena, one of [arenas]. *)
let view arenas p =
  let bytes = match Ops.op p with Bitcast -> Ops.nth p 0 | _ -> p in
  match (Ops.op bytes, Ops.src bytes) with
  | Shrink, [ arena; offset; length ] ->
      Some
        ( arena,
          Bigint.to_int (Shape.to_z offset),
          Bigint.to_int (Shape.to_z length) )
  | Buffer, _ when List.exists (Ops.equal bytes) arenas ->
      Some (bytes, 0, Shape.nbytes bytes)
  | _ -> None

(* The plan of a drawn schedule: its held buffers, its buffers, the plan's
   arenas, and each buffer a call passes or views paired with what the same call
   of the plan passes or views in its place. *)
type plan = {
  held : int list;
  buffers : Ops.t array;
  arenas : Ops.t list;
  places : (int * Ops.t) list;
}

let planned_of d =
  let buffers, linear, held_bufs = build d in
  let planned = Memory.memory_plan_rewrite ~held_bufs linear in
  let unview (a, p) =
    match Ops.op a with
    | Shrink | Bitcast | After -> (Ops.nth a 0, Ops.nth p 0)
    | _ -> (a, p)
  in
  let index b =
    let rec find k = if Ops.equal buffers.(k) b then k else find (k + 1) in
    find 0
  in
  let places =
    List.concat
      (List.map2
         (fun c p ->
           List.map
             (fun ap ->
               let a, p = unview ap in
               (index a, p))
             (List.combine (List.tl (Ops.src c)) (List.tl (Ops.src p))))
         (calls_of linear) (calls_of planned))
  in
  { held = d.held; buffers; arenas = arenas linear planned; places }

(* Each planned buffer's arena, offset and length, once each. *)
let views pl =
  List.sort_uniq compare
    (List.filter_map
       (fun (b, p) ->
         if List.mem b pl.held then None
         else
           Option.map
             (fun (a, off, len) -> (b, (Ops.key a, off, len)))
             (view pl.arenas p))
       pl.places)

(* Each planned buffer's place: its offset less its phase. *)
let places pl =
  List.map
    (fun (b, (_, off, _)) ->
      (b, off - snd (Shape.storage_phase pl.buffers.(b))))
    (views pl)

let held_stay d =
  let pl = planned_of d in
  let held = List.filter (fun (b, _) -> List.mem b d.held) pl.places in
  cover "a call takes a held buffer" (held <> []);
  equal (list uop)
    (List.map (fun (b, _) -> pl.buffers.(b)) held)
    (List.map snd held)

let one_arena d =
  let pl = planned_of d in
  let devices = List.sort_uniq compare (List.map (device d) (planned d)) in
  cover "buffers of two devices are planned" (List.length devices = 2);
  equal
    (list (pair string Dtypes.dtype))
    (List.map (fun dev -> (dev, Dtype.Int8)) devices)
    (List.sort compare
       (List.map (fun a -> (Option.get (Ops.device a), Ops.dtype a)) pl.arenas))

let reach d =
  let pl = planned_of d in
  let reach dev =
    List.fold_left
      (fun r (b, place) ->
        if device d b = dev then max r (place + rounded d b) else r)
      0 (places pl)
  in
  cover "a buffer off 16 bytes is planned"
    (List.exists (fun b -> phase d b <> 0) (planned d));
  equal
    (list (pair string int))
    (List.map
       (fun a ->
         let dev = Option.get (Ops.device a) in
         (dev, reach dev))
       pl.arenas)
    (List.map
       (fun a -> (Option.get (Ops.device a), Shape.max_numel a))
       pl.arenas)

let at_cpu n = (n, Dtype.Int8, 0, cpu)
let passing bs = (List.map (fun b -> (b, Pass)) bs, false)

(* The schedule of the test "buffers are placed largest first". *)
let largest_first =
  {
    sizes = [ at_cpu 256; at_cpu 512; at_cpu 768 ];
    calls = [ passing [ 0 ]; passing [ 0; 1 ]; passing [ 1; 2 ] ];
    depths = [ 0; 0; 0 ];
    held = [];
  }

(* [4] meets [1], at 512, and [3], at 1024, alone: the gaps of 512 bytes at 0
   and of 256 at 768 both hold it, and it takes the smaller. *)
let smaller_gap =
  {
    sizes = [ at_cpu 512; at_cpu 256; at_cpu 256; at_cpu 256; at_cpu 256 ];
    calls = [ passing [ 0; 1; 2; 3 ]; passing [ 1; 3; 4 ]; passing [ 4 ] ];
    depths = [ 0; 0; 0 ];
    held = [];
  }

(* A copy of the held [0] into [1], then a call of [2] and [0]. *)
let copy_then_call =
  {
    sizes = [ at_cpu 256; at_cpu 256; at_cpu 256 ];
    calls = [ ([ (1, Pass); (0, Pass) ], true); passing [ 2; 0 ] ];
    depths = [ 0; 0 ];
    held = [ 0 ];
  }

let placement d =
  let expected = reference d in
  let life = lifetimes d in
  cover "a buffer is placed in a gap"
    (List.exists (fun (_, _, k) -> k >= 1) expected);
  cover "a buffer is placed in the smaller of two gaps that hold it"
    (List.exists (fun (_, _, k) -> k >= 2) expected);
  cover "two buffers of one size on one device meet"
    (List.exists
       (fun (b0, _, _) ->
         List.exists
           (fun (b1, _, _) ->
             b0 < b1
             && rounded d b0 = rounded d b1
             && device d b0 = device d b1
             && meet (life b0) (life b1))
           expected)
       expected);
  equal
    (list (pair int int))
    (List.sort compare (List.map (fun (b, o, _) -> (b, o)) expected))
    (places (planned_of d))

let apart d =
  let pl = planned_of d in
  let life = lifetimes d and views = views pl in
  let meeting =
    List.concat_map
      (fun (b0, (a0, o0, l0)) ->
        List.filter_map
          (fun (b1, (a1, o1, l1)) ->
            if b0 < b1 && a0 = a1 && meet (life b0) (life b1) then
              Some ((b0, b1), ((o0, o0 + l0), (o1, o1 + l1)))
            else None)
          views)
      views
  in
  cover "two buffers whose lives meet share an arena" (meeting <> []);
  cover "a loop of calls" (List.exists (fun k -> k > 0) d.depths);
  let viewed kind =
    List.exists
      (fun (bs, copy) ->
        (not copy)
        && List.exists (fun (b, r) -> r = kind && List.mem_assoc b views) bs)
      d.calls
  in
  cover "a buffer a call shrinks is planned" (viewed Shrink);
  cover "a buffer a call bitcasts is planned" (viewed Bitcast);
  cover "a buffer a call names after a store is planned" (viewed After);
  let copied b =
    List.exists (fun (bs, copy) -> copy && List.mem_assoc b bs) d.calls
  in
  cover "a copy's buffer and another's share an arena"
    (List.exists
       (fun (b0, (a0, _, _)) ->
         List.exists
           (fun (b1, (a1, _, _)) -> a0 = a1 && copied b0 && not (copied b1))
           views)
       views);
  List.iter
    (fun ((b0, b1), spans) ->
      satisfies
        ~msg:(Printf.sprintf "buffers %d and %d" b0 b1)
        ~claim:"bytes apart"
        (pair (pair int int) (pair int int))
        (fun ((s0, e0), (s1, e1)) -> e0 <= s1 || e1 <= s0)
        spans)
    meeting

let as_viewed d =
  let pl = planned_of d in
  cover "a buffer off 16 bytes is planned"
    (List.exists (fun b -> phase d b <> 0) (planned d));
  List.iter
    (fun (b, p) ->
      if not (List.mem b d.held) then begin
        let b = pl.buffers.(b) in
        satisfies ~claim:"a view of an arena's bytes" uop
          (fun p -> Option.is_some (view pl.arenas p))
          p;
        let arena, off, len = Option.get (view pl.arenas p) in
        mem ~msg:"its arena" uop arena pl.arenas;
        let _, phase = Shape.storage_phase b in
        equal ~msg:"its bytes" int (Shape.nbytes b) len;
        equal ~msg:"its phase past a block" int phase (off mod 256);
        equal ~msg:"its alignment and phase" (pair int int)
          (Shape.storage_phase b) (Shape.storage_phase p);
        equal ~msg:"its type" Dtypes.dtype (Ops.dtype b) (Ops.dtype p)
      end)
    pl.places;
  let views = views pl in
  List.iter
    (fun (b, _) ->
      equal
        ~msg:(Printf.sprintf "the places of buffer %d" b)
        (list (triple string int int))
        [ List.assoc b views ]
        (List.map snd (List.filter (fun (b', _) -> b' = b) views)))
    views

let memory_plan_rewrite =
  group "memory_plan_rewrite"
    [
      recorded;
      no_larger;
      prop "a held buffer stays" drawn held_stay;
      test "a schedule of held buffers only is itself" (fun () ->
          equal uop (schedule "all_held") (plan "all_held"));
      test "a schedule on a disk is itself" (fun () ->
          equal uop (schedule "disk") (plan "disk"));
      test "a buffer a call views is planned" (fun () ->
          let a = buf 16 and c = buf 64 in
          let planned =
            Memory.memory_plan_rewrite
              (linear [ call [ c; Shape.shrink a [ Some (Int 0, Int 4) ] ] ])
          in
          equal (list uop) []
            (List.filter
               (fun b ->
                 List.exists (Ops.equal b) (Ops.toposort ~calls:Skip planned))
               [ c; a ]));
      test "a buffer viewed after another's life is not placed over it"
        (fun () ->
          let a = buf 256 and b = buf 256 and x = buf 256 in
          let view = Shape.shrink a [ Some (Int 0, Int 4) ] in
          equal (list int) [ 512 ]
            (arena_bytes ~held_bufs:[ x ]
               (linear [ call [ a; x ]; call [ b; x ]; call [ view; x ] ])));
      test "buffers only one loop's calls take share bytes within a trip"
        (fun () ->
          let x = buf 256 and a = buf 256 and b = buf 256 in
          equal (list int) [ 256 ]
            (arena_bytes ~held_bufs:[ x ]
               (linear [ loop [ call [ a; x ]; call [ b; x ] ] ])));
      test "a buffer a loop's calls and a later call take lives the whole loop"
        (fun () ->
          let x = buf 256 and a = buf 256 and b = buf 256 in
          equal (list int) [ 512 ]
            (arena_bytes ~held_bufs:[ x ]
               (linear [ loop [ call [ a; x ]; call [ b; x ] ]; call [ a; x ] ])));
      (* [c] lives across the inner loop, whose two buffers share. *)
      test "a buffer of a nested loop shares bytes within the outer trip"
        (fun () ->
          let x = buf 256 and a = buf 256 and b = buf 256 and c = buf 256 in
          equal (list int) [ 512 ]
            (arena_bytes ~held_bufs:[ x ]
               (linear
                  [
                    loop
                      [
                        call [ c; x ];
                        loop [ call [ a; x ]; call [ b; x ] ];
                        call [ c; x ];
                      ];
                  ])));
      test "a back edge's flag lives the whole loop" (fun () ->
          let x = buf 256 and a = buf 256 and flag = buf 256 in
          let r = Ops.range ~axis_type:Loop (Int 4) [ Ops.unique_num () ] in
          equal (list int) [ 512 ]
            (arena_bytes ~held_bufs:[ x ]
               (linear
                  [
                    Ops.backedge
                      (linear [ call [ a; x ]; call [ flag; x ] ])
                      ~loop:r ~cond:flag;
                  ])));
      test "a copy's buffer is free from the call after its last" (fun () ->
          let held = buf 256 and c1 = buf 256 and c2 = buf 256 in
          equal (list int) [ 256 ]
            (arena_bytes ~held_bufs:[ held ]
               (linear [ Shape.store_call c1 held; Shape.store_call c2 held ])));
      test
        "a copy's buffer and a call's of one device share bytes once their \
         lives are apart" (fun () ->
          let held = buf 256 and copied = buf 256 and computed = buf 256 in
          equal (list int) [ 256 ]
            (arena_bytes ~held_bufs:[ held ]
               (linear
                  [ Shape.store_call copied held; call [ computed; held ] ])));
      prop "each device has one arena of bytes" drawn one_arena;
      prop
        "an arena is as large as its buffers reach, each its size and phase \
         rounded up to 256 bytes"
        drawn reach;
      prop
        "places buffers largest first, each at the start of the smallest gap \
         that holds it"
        ~count:500
        ~examples:[ largest_first; smaller_gap ]
        drawn placement;
      (* [x] lives over the first two calls, [y] over the last two, [z] over the
         last. Placed as each life starts, [z] would find only [x]'s 256 bytes
         free below [y]; placed largest first, [x] takes bytes that [z], whose
         life it does not meet, takes too, and the arena is the 1280 bytes [y]
         and [z] take at once. *)
      test "buffers are placed largest first" (fun () ->
          let x = buf 256 and y = buf 512 and z = buf 768 in
          equal (list int) [ 1280 ]
            (arena_bytes (linear [ call [ x ]; call [ x; y ]; call [ y; z ] ])));
      prop "buffers whose lives meet share no byte" ~examples:[ copy_then_call ]
        drawn apart;
      prop
        "a planned buffer is its arena's bytes from its phase past its place, \
         viewed as its type"
        drawn as_viewed;
      test "new arenas take the next slots" (fun () ->
          let linear = schedule "simple" in
          let before = Ops.unique_num () in
          satisfies
            ~claim:(Printf.sprintf "one slot past %d" before)
            (list int)
            (function [ s ] -> s > before | _ -> false)
            (List.map
               (fun a -> (Ops.param_arg_of a).slot)
               (arenas linear (Memory.memory_plan_rewrite linear))));
      test "without the planner, a schedule is itself" (fun () ->
          let linear = schedule "simple" in
          equal uop linear
            (Setting.context
               [ Setting.B (Setting.no_memory_planner, true) ]
               (fun () -> Memory.memory_plan_rewrite linear)));
      printouts;
      test "below debug level 1 prints nothing" (fun () ->
          Setting.context
            [ Setting.B (Setting.debug, 0) ]
            (fun () -> ignore (plan "sizes"));
          equal string "" (output ()));
    ]

(* buffers, bytes, taken *)

let param () =
  Shape.param ~device:cpu ~shape:[ Int 4 ] (Ops.unique_num ()) Float32

let alloc () =
  Ops.v Alloc
    ~src:[ Ops.size_src (Some 4) ]
    ~arg:(Param (Ops.param_arg ~slot:(Ops.unique_num ()) ~device:cpu Float32))

let backedge entries flag =
  Ops.backedge (linear entries)
    ~loop:(Ops.range ~axis_type:Loop (Int 4) [ Ops.unique_num () ])
    ~cond:flag

let buffers =
  group "buffers"
    [
      test
        "is each buffer a loop's calls pass or view, once, in order, without \
         call-local storage or parameters" (fun () ->
          let a = buf 16 and b = buf 32 and c = buf 64 in
          equal (list uop) [ a; c; b ]
            (Memory.buffers
               (loop
                  [
                    call [ a; alloc (); c ];
                    call [ param (); Shape.shrink b [ Some (Int 0, Int 4) ]; a ];
                  ])));
      test "holds a back edge's flag" (fun () ->
          let a = buf 16 and flag = buf 4 in
          mem uop flag (Memory.buffers (backedge [ call [ a ] ] flag)));
    ]

(* Each expected value is the row's size plus phase rounded up to a multiple of
   256, as bytes states. *)
let bytes =
  group "bytes"
    [
      cases "is a buffer's size and phase, rounded up to 256"
        ~id:(fun (id, _, _) -> id)
        [
          ("1 byte at phase 0", (1, Dtype.Int8, 0), 256);
          ("256 bytes at phase 0", (256, Int8, 0), 256);
          ("257 bytes at phase 0", (257, Int8, 0), 512);
          ("252 bytes at phase 4", (252, Int8, 4), 256);
          ("253 bytes at phase 4", (253, Int8, 4), 512);
          ("128 Float16s at phase 0", (128, Float16, 0), 256);
        ]
        (fun (_, (n, dt, phase), expected) ->
          equal int expected (Memory.bytes (buf ~dt ~phase n)));
    ]

let taken =
  group "taken"
    [
      test "of a loop is each storage its calls pass or view, once, in order"
        (fun () ->
          let a = buf 16 and b = buf 32 and c = buf 64 in
          equal (list uop) [ a; c; b ]
            (Memory.taken
               (loop
                  [
                    call [ a; c ];
                    call [ Shape.shrink b [ Some (Int 0, Int 4) ]; a ];
                  ])));
      test "holds call-local storage, parameters and the buffers views reach"
        (fun () ->
          let al = alloc () and p = param () and b = buf 32 in
          equal (list uop) [ al; p; b ]
            (Memory.taken
               (call [ al; p; Shape.shrink b [ Some (Int 0, Int 4) ] ])));
      test "holds a back edge's flag" (fun () ->
          let a = buf 16 and flag = buf 4 in
          mem uop flag (Memory.taken (backedge [ call [ a ] ] flag)));
    ]

let () = exit (run "Tolk.Memory" [ memory_plan_rewrite; buffers; bytes; taken ])
