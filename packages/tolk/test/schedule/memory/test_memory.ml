(* Tests of Tolk.Memory: the plans of tinygrad's schedules, the buffers a call
   takes, and the laws of a plan over generated schedules. *)

open Windtrap
open Tolk

let uop = Uops.uop
let cpu = "CPU"

(* Plans *)

(* The storage [u] makes that [linear] does not hold, the arenas. *)
let arenas linear u =
  let held = Ops.backward_slice_with_self ~calls:Skip linear in
  List.filter
    (fun n -> Ops.op n = Buffer && not (Ops.Nodes.mem n held))
    (Ops.toposort ~calls:Enter u)

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

let cases =
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
  group "memory_plan_rewrite › recorded"
    (List.map
       (fun name ->
         let file = name ^ "_planned.golden" in
         Golden.graph file (fun () ->
             numbered_as (Golden.sink file) (schedule name) (plan name)))
       cases)

(* Each recorded schedule's arenas take no more bytes, summed, than
   tinygrad's plan of it: the placement of D477 against its TLSF. *)
let no_larger =
  group "memory_plan_rewrite › arenas no larger than tinygrad's"
    (List.map
       (fun name ->
         test name (fun () ->
             let linear = schedule name in
             let bytes u =
               List.fold_left
                 (fun n a -> n + Shape.max_numel a)
                 0 (arenas linear u)
             in
             let upstream = Golden.upstream (name ^ "_planned.golden") in
             at_most int ~than:(bytes upstream) (bytes (plan name))))
       cases)

let chomp s =
  if String.ends_with ~suffix:"\n" s then String.sub s 0 (String.length s - 1)
  else s

let printouts =
  group "memory_plan_rewrite › debug"
    (List.map
       (fun name ->
         Golden.text (name ^ "_debug.golden") (fun () ->
             Setting.context
               [ Setting.B (Setting.debug, 1) ]
               (fun () -> ignore (plan name));
             chomp (output ())))
       [ "simple"; "sizes"; "copies" ]
    @ [
        test "below debug level 1, nothing is printed" (fun () ->
            ignore (plan "sizes");
            equal string "" (output ()));
      ])

(* The buffers a call takes *)

let buf ?(device = cpu) ?(dt = Dtype.Int8) ?phase size =
  Ops.new_buffer ?phase device size dt

let reached u = Ops.toposort ~calls:Skip u
let planned_away u b = not (List.exists (( == ) b) (reached u))

let taken =
  let a = buf 16 and c = buf 64 in
  let run args = Ops.v Linear ~src:[ Ops.call (Ops.sink args) args ] in
  group "memory_plan_rewrite › the buffers a call takes"
    [
      test "an entry takes each buffer its calls pass or view, once" (fun () ->
          let b = buf 32 in
          let si =
            Ops.v Linear
              ~src:[ Ops.call (Ops.sink [ a; c ]) [ a; c ]; Ops.call (Ops.sink [ b; a ]) [ Shape.shrink b [ Some (Int 0, Int 4) ]; a ] ]
          in
          let r = Ops.range ~axis_type:Loop (Int 2) [ 0 ] in
          equal (list uop) [ a; c; b ] (Memory.taken (Ops.end_ si [ r ])));
      test "a buffer a call views is planned" (fun () ->
          let linear = run [ c; Shape.shrink a [ Some (Int 0, Int 4) ] ] in
          let planned = Memory.memory_plan_rewrite linear in
          equal (list bool) [ true; true ]
            (List.map (planned_away planned) [ c; a ]));
      test "a buffer viewed after another's life is not placed over it"
        (fun () ->
          let a = buf 256 and b = buf 256 and x = buf 256 in
          let view = Shape.shrink a [ Some (Int 0, Int 4) ] in
          let call args = Ops.call (Ops.sink args) args in
          let linear =
            Ops.v Linear ~src:[ call [ a; x ]; call [ b; x ]; call [ view; x ] ]
          in
          let planned = Memory.memory_plan_rewrite ~held_bufs:[ x ] linear in
          equal (list int) [ 512 ]
            (List.map Shape.max_numel (arenas linear planned)));
    ]

(* Loops *)

let loops =
  let call args = Ops.call (Ops.sink args) args in
  let loop calls =
    Ops.end_
      (Ops.v Linear ~src:calls)
      [ Ops.range ~axis_type:Loop (Int 4) [ Ops.unique_num () ] ]
  in
  let bytes linear held_bufs =
    List.map Shape.max_numel
      (arenas linear (Memory.memory_plan_rewrite ~held_bufs linear))
  in
  group "memory_plan_rewrite › loops"
    [
      test "buffers only one loop's calls take share bytes within a trip"
        (fun () ->
          let x = buf 256 and a = buf 256 and b = buf 256 in
          let linear = Ops.v Linear ~src:[ loop [ call [ a; x ]; call [ b; x ] ] ] in
          equal (list int) [ 256 ] (bytes linear [ x ]));
      test "a buffer a loop's calls and a later call take lives the whole loop"
        (fun () ->
          let x = buf 256 and a = buf 256 and b = buf 256 in
          let linear =
            Ops.v Linear
              ~src:[ loop [ call [ a; x ]; call [ b; x ] ]; call [ a; x ] ]
          in
          equal (list int) [ 512 ] (bytes linear [ x ]));
      test "a buffer of a nested loop shares bytes within the outer trip"
        (fun () ->
          let x = buf 256 and a = buf 256 and b = buf 256 and c = buf 256 in
          let linear =
            Ops.v Linear
              ~src:
                [
                  loop
                    [
                      call [ c; x ];
                      loop [ call [ a; x ]; call [ b; x ] ];
                      call [ c; x ];
                    ];
                ]
          in
          (* [c] lives across the inner loop, whose two buffers share. *)
          equal (list int) [ 512 ] (bytes linear [ x ]));
      test "a back edge's flag lives the whole loop" (fun () ->
          let x = buf 256 and a = buf 256 and flag = buf 256 in
          let r = Ops.range ~axis_type:Loop (Int 4) [ Ops.unique_num () ] in
          let linear =
            Ops.v Linear
              ~src:
                [
                  Ops.backedge
                    (Ops.v Linear ~src:[ call [ a; x ]; call [ flag; x ] ])
                    ~loop:r ~cond:flag;
                ]
          in
          equal (list int) [ 512 ] (bytes linear [ x ]));
    ]

(* Rules *)

let rules =
  group "memory_plan_rewrite › rules"
    [
      test "without the planner, a schedule is itself" (fun () ->
          let linear = schedule "simple" in
          equal uop linear
            (Setting.context
               [ Setting.B (Setting.no_memory_planner, true) ]
               (fun () -> Memory.memory_plan_rewrite linear)));
      test "a schedule of held buffers only is itself" (fun () ->
          equal uop (schedule "all_held") (plan "all_held"));
      test "each plan's arenas take new slots" (fun () ->
          let linear = schedule "simple" in
          let slots () =
            List.map
              (fun a -> match Ops.arg a with Param p -> p.slot | _ -> -1)
              (arenas linear (Memory.memory_plan_rewrite linear))
          in
          let first = slots () and second = slots () in
          equal int 1 (List.length first);
          is_true ~msg:"the second plan's slots are new"
            (List.for_all (fun s -> not (List.mem s first)) second));
      test "a schedule on a disk is itself" (fun () ->
          equal uop (schedule "disk") (plan "disk"));
      (* [x] lives over the first two calls, [y] over the last two, [z] over
         the last. Placed as each life starts, [z] would find only [x]'s 256
         bytes free below [y]; placed largest first, [x] takes bytes that [z],
         whose life it does not meet, takes too, and the arena is the 1280
         bytes [y] and [z] take at once. *)
      test "buffers are placed largest first" (fun () ->
          let x = buf 256 and y = buf 512 and z = buf 768 in
          let call args = Ops.call (Ops.sink args) args in
          let linear =
            Ops.v Linear ~src:[ call [ x ]; call [ x; y ]; call [ y; z ] ]
          in
          equal (list int) [ 1280 ]
            (List.map Shape.max_numel
               (arenas linear (Memory.memory_plan_rewrite linear))));
      test "a copy's buffer is free from the call after its last" (fun () ->
          let held = buf 256 and c1 = buf 256 and c2 = buf 256 in
          let linear =
            Ops.v Linear
              ~src:[ Shape.store_call c1 held; Shape.store_call c2 held ]
          in
          let planned = Memory.memory_plan_rewrite ~held_bufs:[ held ] linear in
          equal (list int) [ 256 ]
            (List.map Shape.max_numel (arenas linear planned)));
      test
        "a copy's buffer and a call's of one device share bytes once their \
         lives are apart" (fun () ->
          let held = buf 256 and copied = buf 256 and computed = buf 256 in
          let linear =
            Ops.v Linear
              ~src:
                [
                  Shape.store_call copied held;
                  Ops.call (Ops.sink [ computed; held ]) [ computed; held ];
                ]
          in
          let planned =
            Memory.memory_plan_rewrite ~held_bufs:[ held ] linear
          in
          equal (list int) [ 256 ]
            (List.map Shape.max_numel (arenas linear planned)));
    ]

(* Laws

   Over generated schedules of calls of buffers, some calls copies between two,
   some arguments views of a buffer, some buffers held: a planned buffer is the
   bytes of an arena at its place,
   viewed as its type and at its phase; two buffers whose lifetimes meet never
   share a byte; the device has one arena, copies' buffers and the others'
   alike, of at least the bytes its buffers live at once, each rounded as the
   plan rounds it; held buffers stay. A buffer lives from the first call that
   takes it, passed or viewed, to the last. *)

type drawn = {
  sizes : (int * Dtype.t * int) list;
  calls : ((int * bool) list * bool) list;
  depths : int list;
      (** Each call's loop depth: consecutive calls of depth 1 or more run in
          one loop, and among them consecutive calls of depth 2 in a loop
          inside it. *)
  held : int list;
}

let pp_drawn ppf d =
  let pp_arg (b, view) =
    if view then Printf.sprintf "view %d" b else string_of_int b
  in
  let pp_call ppf (bs, copy) =
    Format.fprintf ppf "%s(%s)"
      (if copy then "copy" else "call")
      (String.concat ", " (List.map pp_arg bs))
  in
  Format.fprintf ppf "sizes [%s], %a, depths [%s], held [%s]"
    (String.concat "; "
       (List.map
          (fun (n, dt, p) -> Format.asprintf "%d %a at %d" n Dtype.pp dt p)
          d.sizes))
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
      (triple
         (of_list [ 1; 16; 255; 256; 257; 1000; 4096; 70_000 ])
         (of_list [ Dtype.Int8; Float32; Float16 ])
         (of_list [ 0; 0; 4; 12 ]))
  and+ calls, depths =
    let* calls =
      list ~size:(int_range 1 12)
        (let* bs = list ~size:(int_range 1 4) (pair buffer bool) in
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

let build d =
  let buffers =
    Array.of_list (List.map (fun (n, dt, phase) -> buf ~dt ~phase n) d.sizes)
  in
  (* A call passes a buffer, or views its first element. *)
  let arg (i, view) =
    let b = buffers.(i) in
    if view then Shape.shrink b [ Some (Int 0, Int 1) ] else b
  in
  let calls =
    List.map
      (fun (bs, copy) ->
        if copy then
          let dst, src = (fst (List.hd bs), fst (List.nth bs 1)) in
          Shape.store_call buffers.(dst) buffers.(src)
        else
          let args = List.map arg bs in
          Ops.call (Ops.sink args) args)
      d.calls
  in
  (* Consecutive calls at depth [k] or more run in one loop of trips around
     those at depth [k + 1] or more. *)
  let loop entries =
    Ops.end_
      (Ops.v Linear ~src:entries)
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
  (buffers, Ops.v Linear ~src:entries, List.map (fun i -> buffers.(i)) d.held)

(* The calls of [linear], those of its loops in their places. *)
let rec calls_of linear =
  List.concat_map
    (fun e ->
      if Ops.op e = End then calls_of (Ops.nth e 0) else [ e ])
    (Ops.src linear)

(* [placed linear planned] is each buffer a call of [linear] passes or views,
   with what the same call of [planned] passes or views in its place. *)
let placed linear planned =
  let unview (a, p) =
    if Ops.op a = Shrink then (Ops.nth a 0, Ops.nth p 0) else (a, p)
  in
  List.concat
    (List.map2
       (fun c p ->
         List.map unview
           (List.combine (List.tl (Ops.src c)) (List.tl (Ops.src p))))
       (calls_of linear) (calls_of planned))

let view u =
  let bytes = match Ops.op u with Bitcast -> Ops.nth u 0 | _ -> u in
  match Ops.src bytes with
  | [ arena; offset; length ] when Ops.op bytes = Shrink ->
      Some
        (arena, Bigint.to_int (Shape.to_z offset), Bigint.to_int (Shape.to_z length))
  | _ -> None

(* Each buffer's life, in calls: from the first call that takes it to the
   last, where every call that takes it lies in one loop; otherwise each use
   inside a loop that not all its uses lie in holds it across that loop's
   calls, the outermost such loop. Each call's loops are named by the
   number of the call that opens them. *)
let lifetimes d =
  let depths = Array.of_list d.depths in
  let n = Array.length depths in
  (* The calls of the loop at depth [k] around call [i]: its first and last. *)
  let span i k =
    let lo = ref i and hi = ref i in
    while !lo > 0 && depths.(!lo - 1) >= k do decr lo done;
    while !hi < n - 1 && depths.(!hi + 1) >= k do incr hi done;
    (!lo, !hi)
  in
  let uses = Hashtbl.create 8 and copies = Hashtbl.create 8 in
  List.iteri
    (fun i (bs, copy) ->
      List.iter
        (fun (b, _) ->
          Hashtbl.replace uses b
            (i :: Option.value ~default:[] (Hashtbl.find_opt uses b));
          if copy then Hashtbl.replace copies b ())
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
    (lo, hi + 1, Hashtbl.mem copies b)

let plans d =
  let buffers, linear, held_bufs = build d in
  let planned = Memory.memory_plan_rewrite ~held_bufs linear in
  let places = placed linear planned in
  let index b =
    let rec find k = if Ops.equal buffers.(k) b then k else find (k + 1) in
    find 0
  in
  let life = lifetimes d in
  let views =
    List.sort_uniq compare
      (List.filter_map
         (fun (b, p) ->
           Option.map
             (fun (arena, off, len) -> (index b, (Ops.key arena, off, len)))
             (view p))
         places)
  in
  (* Held buffers stay, and a buffer is planned once, at one place. *)
  List.iter
    (fun (b, p) ->
      if List.exists (Ops.equal b) held_bufs then
        equal ~msg:"a held buffer stays" uop b p)
    places;
  List.iter
    (fun (b, place) ->
      List.iter
        (fun (b', place') ->
          if b = b' then
            equal ~msg:"one place per buffer" (triple string int int) place
              place')
        views)
    views;
  (* Each place is its buffer's bytes, viewed as its type. *)
  List.iter
    (fun (b, p) ->
      match view p with
      | Some (_, off, len) ->
          let _, phase = Shape.storage_phase b in
          equal ~msg:"a view of the buffer's bytes" int (Shape.nbytes b) len;
          equal ~msg:"a place its phase past a block" int 0
            ((off - phase) mod 256);
          equal ~msg:"a view at the buffer's phase" (pair int int)
            (Shape.storage_phase b) (Shape.storage_phase p);
          equal ~msg:"a view of the buffer's type" Dtypes.dtype (Ops.dtype b)
            (Ops.dtype p)
      | None -> ())
    places;
  (* Buffers that live at once share no byte, and the device has one arena,
     of at least the bytes its buffers live at once, each rounded up to 256
     bytes from its phase. *)
  let arenas = arenas linear planned in
  at_most int ~msg:"the device's arenas" ~than:1 (List.length arenas);
  let rounded b =
    let _, phase = Shape.storage_phase buffers.(b) in
    (phase + Shape.nbytes buffers.(b) + 255) / 256 * 256
  in
  let planned_bufs = List.sort_uniq compare (List.map fst views) in
  let live_at t =
    List.fold_left
      (fun n b ->
        let s, e, _ = life b in
        if s <= t && t < e then n + rounded b else n)
      0 planned_bufs
  in
  let peak =
    List.fold_left max 0 (List.init (List.length d.calls) live_at)
  in
  (match arenas with
  | [ arena ] ->
      at_least int ~msg:"the arena's bytes" ~than:peak (Shape.max_numel arena)
  | _ -> equal int ~msg:"no planned buffer" 0 peak);
  List.iter
    (fun (b0, (a0, o0, l0)) ->
      List.iter
        (fun (b1, (a1, o1, l1)) ->
          if b0 < b1 && a0 = a1 then begin
            let s0, e0, _ = life b0 and s1, e1, _ = life b1 in
            if s0 < e1 && s1 < e0 then
              is_true
                ~msg:
                  (Printf.sprintf "buffers %d and %d live at once and overlap"
                     b0 b1)
                (o0 + l0 <= o1 || o1 + l1 <= o0)
          end)
        views)
    views;
  cover "buffers share an arena" (List.length views > 1);
  cover "a copy's buffer and a call's share an arena"
    (List.exists
       (fun (b0, (a0, _, _)) ->
         List.exists
           (fun (b1, (a1, _, _)) ->
             a0 = a1
             &&
             let _, _, c0 = life b0 and _, _, c1 = life b1 in
             c0 <> c1)
           views)
       views);
  cover "a buffer a call views is planned"
    (List.exists
       (fun (bs, copy) ->
         (not copy)
         && List.exists
              (fun (b, view) -> view && List.mem_assoc b views)
              bs)
       d.calls);
  cover "a loop of calls" (List.exists (fun k -> k > 0) d.depths);
  cover "a buffer off 16 bytes is planned"
    (List.exists
       (fun (b, _) -> snd (Shape.storage_phase buffers.(b)) <> 0)
       views)

let laws =
  group "memory_plan_rewrite › laws"
    [ prop "a plan places live buffers apart, each at its bytes" drawn plans ]

let () =
  exit
    (run "Tolk.Memory"
       [ recorded; no_larger; printouts; taken; loops; rules; laws ])
