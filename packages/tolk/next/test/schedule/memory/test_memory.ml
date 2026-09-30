(* Tests of Tolk_next.Memory: the plans of tinygrad's schedules, the buffers a
   call passes, and the laws of a plan over generated schedules. *)

open Windtrap
open Tolk_next

let uop = Uops.uop
let cpu = Ops.Single "CPU"

(* Plans *)

(* The storage [u] makes that [linear] does not hold, the arenas. *)
let arenas linear u =
  let held = Ops.backward_slice_with_self linear in
  List.filter
    (fun n -> Ops.op n = Buffer && not (Ops.Nodes.mem n held))
    (Ops.toposort u)

(* [numbered_as golden linear u] is [u] with its arenas numbered as [golden]'s:
   new storage takes its number from a counter the process shares, so the
   numbers are no property of the plan. *)
let numbered_as golden linear u =
  let slot n =
    match Ops.arg n with Param p -> p | _ -> fail "an arena has a parameter"
  in
  Ops.substitute u
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
    "two_devices";
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
    "sharded";
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
             Helpers.context
               [ Helpers.B (Helpers.debug, 1) ]
               (fun () -> ignore (plan name));
             chomp (output ())))
       [ "simple"; "sizes"; "copies" ]
    @ [
        test "below debug level 1, nothing is printed" (fun () ->
            ignore (plan "sizes");
            equal string "" (output ()));
      ])

(* collect_bufs *)

let buf ?(device = cpu) ?(dt = Dtype.Int8) size = Ops.new_buffer device size dt

let collected =
  let two = Ops.Multi [ "CPU:0"; "CPU:1" ] in
  let a = buf 16 and b = buf 32 and m = buf ~device:two 16 in
  group "collect_bufs"
    [
      test "a buffer is itself" (fun () ->
          equal (list uop) [ a ] (Memory.collect_bufs a));
      test "a shard selection and a gather pass their sources' buffers"
        (fun () ->
          equal (list uop) [ m ] (Memory.collect_bufs (Ops.mselect m 1));
          equal (list uop) [ a; b; a; a ]
            (Memory.collect_bufs
               (Ops.mstack a [ Ops.mselect (Ops.mstack b [ a ]) 0; a ])));
      test "a view of a buffer passes none" (fun () ->
          equal (list uop) []
            (Memory.collect_bufs (Ops.shrink a [ Some (Int 0, Int 4) ])));
    ]

(* Rules *)

let rules =
  group "memory_plan_rewrite › rules"
    [
      test "without the planner, a schedule is itself" (fun () ->
          let linear = schedule "simple" in
          equal uop linear
            (Helpers.context
               [ Helpers.B (Helpers.no_memory_planner, true) ]
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
    ]

(* Laws

   Over generated schedules of calls of buffers, some calls copies between two,
   some buffers held: a planned buffer is the bytes of an arena at its place,
   viewed as its type; two buffers whose lifetimes meet never share a byte;
   copies and the other calls use arenas of their own; held buffers stay. A
   buffer lives from the first call that takes it to the last, and a copy's
   buffers for as many calls again after. *)

type drawn = {
  sizes : (int * Dtype.t) list;
  calls : (int list * bool) list;
  held : int list;
}

let pp_drawn ppf d =
  let pp_call ppf (bs, copy) =
    Format.fprintf ppf "%s(%s)"
      (if copy then "copy" else "call")
      (String.concat ", " (List.map string_of_int bs))
  in
  Format.fprintf ppf "sizes [%s], %a, held [%s]"
    (String.concat "; "
       (List.map (fun (n, dt) -> Format.asprintf "%d %a" n Dtype.pp dt) d.sizes))
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
      (pair
         (of_list [ 1; 16; 255; 256; 257; 1000; 4096; 70_000 ])
         (of_list [ Dtype.Int8; Float32; Float16 ]))
  and+ calls =
    list ~size:(int_range 1 12)
      (let* bs = list ~size:(int_range 1 4) buffer in
       let+ copy = bool in
       (bs, copy && List.length bs = 2 && List.hd bs <> List.nth bs 1))
  and+ held = list ~size:(int_range 0 3) buffer in
  { sizes; calls; held }

let drawn = Gen.with_pp pp_drawn drawn

let build d =
  let buffers = Array.of_list (List.map (fun (n, dt) -> buf ~dt n) d.sizes) in
  let calls =
    List.map
      (fun (bs, copy) ->
        let bs = List.map (fun i -> buffers.(i)) bs in
        if copy then Ops.store_call (List.hd bs) (List.nth bs 1)
        else Ops.call (Ops.sink bs) bs)
      d.calls
  in
  (buffers, Ops.v Linear ~src:calls, List.map (fun i -> buffers.(i)) d.held)

(* [placed linear planned] is each buffer a call of [linear] passes, with what
   the same call of [planned] passes in its place. *)
let placed linear planned =
  List.concat
    (List.map2
       (fun c p -> List.combine (List.tl (Ops.src c)) (List.tl (Ops.src p)))
       (Ops.src linear) (Ops.src planned))

let view u =
  let bytes = match Ops.op u with Bitcast -> Ops.nth u 0 | _ -> u in
  match Ops.src bytes with
  | [ arena; offset; length ] when Ops.op bytes = Shrink ->
      Some (arena, Z.to_int (Ops.to_z offset), Z.to_int (Ops.to_z length))
  | _ -> None

let lifetimes d =
  let first = Hashtbl.create 8 and last = Hashtbl.create 8 in
  let copies = Hashtbl.create 8 in
  List.iteri
    (fun i (bs, copy) ->
      List.iter
        (fun b ->
          if not (Hashtbl.mem first b) then Hashtbl.replace first b i;
          Hashtbl.replace last b i;
          if copy then Hashtbl.replace copies b ())
        bs)
    d.calls;
  fun b ->
    let f = Hashtbl.find first b and l = Hashtbl.find last b in
    let hold = if Hashtbl.mem copies b then l - f + 1 else 0 in
    (f, l + 1 + hold, Hashtbl.mem copies b)

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
          equal ~msg:"a view of the buffer's bytes" int (Ops.nbytes b) len;
          equal ~msg:"a place on a block" int 0 (off mod 256);
          equal ~msg:"a view of the buffer's type" Dtypes.dtype (Ops.dtype b)
            (Ops.dtype p)
      | None -> ())
    places;
  (* Buffers that live at once share no byte, and copies share no arena with
     other calls. *)
  List.iter
    (fun (b0, (a0, o0, l0)) ->
      List.iter
        (fun (b1, (a1, o1, l1)) ->
          if b0 < b1 && a0 = a1 then begin
            let s0, e0, c0 = life b0 and s1, e1, c1 = life b1 in
            is_true
              ~msg:
                (Printf.sprintf
                   "buffers %d and %d share an arena, copy and call" b0 b1)
              (c0 = c1);
            if s0 < e1 && s1 < e0 then
              is_true
                ~msg:
                  (Printf.sprintf "buffers %d and %d live at once and overlap"
                     b0 b1)
                (o0 + l0 <= o1 || o1 + l1 <= o0)
          end)
        views)
    views;
  cover "buffers share an arena" (List.length views > 1)

let laws =
  group "memory_plan_rewrite › laws"
    [ prop "a plan places live buffers apart, each at its bytes" drawn plans ]

let () = exit (run "Memory" [ recorded; printouts; collected; rules; laws ])
