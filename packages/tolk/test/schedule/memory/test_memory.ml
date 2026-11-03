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
  Format.fprintf ppf "sizes [%s], %a, held [%s]"
    (String.concat "; "
       (List.map
          (fun (n, dt, p) -> Format.asprintf "%d %a at %d" n Dtype.pp dt p)
          d.sizes))
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       pp_call)
    d.calls
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
  and+ calls =
    list ~size:(int_range 1 12)
      (let* bs = list ~size:(int_range 1 4) (pair buffer bool) in
       let+ copy = bool in
       let copy =
         copy && List.length bs = 2 && fst (List.hd bs) <> fst (List.nth bs 1)
       in
       (bs, copy))
  and+ held = list ~size:(int_range 0 3) buffer in
  { sizes; calls; held }

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
  (buffers, Ops.v Linear ~src:calls, List.map (fun i -> buffers.(i)) d.held)

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
       (Ops.src linear) (Ops.src planned))

let view u =
  let bytes = match Ops.op u with Bitcast -> Ops.nth u 0 | _ -> u in
  match Ops.src bytes with
  | [ arena; offset; length ] when Ops.op bytes = Shrink ->
      Some
        (arena, Bigint.to_int (Shape.to_z offset), Bigint.to_int (Shape.to_z length))
  | _ -> None

let lifetimes d =
  let first = Hashtbl.create 8 and last = Hashtbl.create 8 in
  let copies = Hashtbl.create 8 in
  List.iteri
    (fun i (bs, copy) ->
      List.iter
        (fun (b, _) ->
          if not (Hashtbl.mem first b) then Hashtbl.replace first b i;
          Hashtbl.replace last b i;
          if copy then Hashtbl.replace copies b ())
        bs)
    d.calls;
  fun b ->
    (Hashtbl.find first b, Hashtbl.find last b + 1, Hashtbl.mem copies b)

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
  cover "a buffer off 16 bytes is planned"
    (List.exists
       (fun (b, _) -> snd (Shape.storage_phase buffers.(b)) <> 0)
       views)

let laws =
  group "memory_plan_rewrite › laws"
    [ prop "a plan places live buffers apart, each at its bytes" drawn plans ]

let () =
  exit (run "Tolk.Memory" [ recorded; printouts; taken; rules; laws ])
