(* Tests of Tolk.Allreduce: the allreduces tinygrad expands, each algorithm's
   choice, and the law that an expanded allreduce computes the reduction it
   expands. *)

open Windtrap
open Tolk

let uop = Uops.uop

(* Cases

   Each case of the generator, with its settings. *)

let ring n = Setting.B (Setting.ring, n)
let all2all n = Setting.B (Setting.all2all, n)
let nodes n = Setting.B (Setting.allreduce_node_ndevs, n)

let cases =
  [
    ("naive_two_devices", []);
    ("naive_to_one_device", []);
    ("naive_four_devices", []);
    ("naive_int", []);
    ("naive_two_devices_many_elements", []);
    ("naive_symbolic", [ ring 2; all2all 2; nodes 2 ]);
    ("ring_two_devices", [ ring 2 ]);
    ("ring_uneven_chunks", [ ring 2 ]);
    ("ring_chunks_of_eight", [ ring 2 ]);
    ("ring_to_one_device", [ ring 2 ]);
    ("ring_many_elements", []);
    ("ring_off", [ ring 0 ]);
    ("all2all", [ all2all 2; ring 2 ]);
    ("all2all_to_one_device", [ all2all 2 ]);
    ("all2all_many_elements", [ all2all 1 ]);
    ("nodes_of_two", [ nodes 2 ]);
    ("nodes_of_three", [ nodes 3 ]);
    ("nodes_to_one_device", [ nodes 2 ]);
    ("nodes_not_dividing", [ nodes 2 ]);
  ]

let settings name = List.assoc name cases
let allreduce name = Ops.nth (Golden.sink (name ^ ".golden")) 0

let handled ?(settings = []) red =
  Setting.context settings (fun () ->
      require_some (Allreduce.handle_allreduce red))

let created ?(settings = []) red =
  Setting.context settings (fun () -> Allreduce.create_allreduce_function red)

(* The storage a graph makes that [red] does not hold. *)
let made_storage red u =
  let held = Ops.backward_slice_with_self ~calls:Skip red in
  List.filter
    (fun n -> Ops.op n = Alloc && not (Ops.Nodes.mem n held))
    (Ops.toposort ~calls:Enter u)

(* [numbered_as golden red u] is [u] with the storage it makes numbered as the
   storage [golden] makes: numbers of new storage come from a counter that the
   process shares, so they are no property of the function. *)
let numbered_as golden red u =
  let slot n =
    match Ops.arg n with Param p -> p | _ -> fail "storage has a parameter"
  in
  let subs =
    List.map2
      (fun mine theirs ->
        ( mine,
          Ops.replace
            ~arg:(Param { (slot mine) with slot = (slot theirs).slot })
            mine ))
      (made_storage red u) (made_storage red golden)
  in
  Ops.substitute ~calls:Skip u subs

(* Tinygrad gathers each chunk a hierarchical allreduce reduces on every
   device, whatever its target. Where the target is one device, tolk copies
   there the chunk device [k] of the first node reduced, which each gather holds
   first. *)
let landed_on_its_device golden =
  let target =
    match Ops.arg (allreduce "nodes_to_one_device") with
    | Allreduce { device; _ } -> device
    | _ -> fail "an allreduce has a target"
  in
  let gathers =
    List.filter (fun n -> Ops.op n = Mstack) (Ops.toposort ~calls:Enter golden)
  in
  Ops.substitute ~calls:Skip golden
    (List.map
       (fun m -> (m, Ops.copy_to_device (Ops.nth (Ops.nth m 0) 0) target))
       gathers)

let expansions =
  let recorded (name, settings) =
    let file = name ^ "_handled.golden" in
    let handled () = Ops.sink [ handled ~settings (allreduce name) ] in
    if name = "nodes_to_one_device" then
      test (file ^ ", landed on its device") (fun () ->
          equal uop (landed_on_its_device (Golden.sink file)) (handled ()))
    else Golden.graph file handled
  in
  group "handle_allreduce › recorded" (List.map recorded cases)

let functions =
  group "create_allreduce_function › recorded"
    (List.map
       (fun name ->
         let file = name ^ "_function.golden" in
         Golden.graph file (fun () ->
             let red = allreduce name in
             numbered_as
               (Ops.nth (Golden.sink file) 0)
               red
               (Ops.sink [ created ~settings:(settings name) red ])))
       [
         "naive_two_devices";
         "naive_symbolic";
         "ring_uneven_chunks";
         "ring_to_one_device";
         "nodes_of_two";
       ])

let chomp s =
  if String.ends_with ~suffix:"\n" s then String.sub s 0 (String.length s - 1)
  else s

let printouts =
  group "handle_allreduce › debug"
    (List.map
       (fun name ->
         Golden.text (name ^ "_debug.golden") (fun () ->
             ignore
               (handled
                  ~settings:(Setting.B (Setting.debug, 2) :: settings name)
                  (allreduce name));
             chomp (output ())))
       [
         "naive_two_devices"; "naive_symbolic"; "ring_uneven_chunks"; "all2all";
       ]
    @ [
        test "below debug level 2, nothing is printed" (fun () ->
            ignore
              (handled
                 ~settings:[ Setting.B (Setting.debug, 1) ]
                 (allreduce "naive_two_devices"));
            equal string "" (output ()));
      ])

(* Values

   The law that an expanded allreduce computes the reduction: on each device it
   places its value on, the expansion holds the elementwise reduction of the
   source's shards (Tensors), where each shard holds small integers, so that
   sums are exact in any order. *)

let devices = function Some (Ops.Multi l) -> List.length l | _ -> 1

let filled u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | Buffer, Param { slot; size = Some size; dtype; device; _ } ->
          let element j : Dtype.value =
            let k = (((j * 7) + (slot * 3)) mod 11) - 3 in
            if Dtype.is_float dtype then `Float (float_of_int k)
            else `Int (Bigint.of_int k)
          in
          Some (slot, Array.init (size * devices device) element)
      | _ -> None)
    (Ops.toposort ~calls:Enter u)

let value red u = Tensors.eval ~buffers:(filled red) u

(* [holds_reduction red u] checks that each device of [u] holds the reduction of
   [red]'s shards, and that [u] is on [red]'s devices. *)
let holds_reduction red u =
  let reduction = value red red in
  let devices = value red u in
  List.iteri
    (fun k v ->
      equal
        ~msg:(Printf.sprintf "device %d" k)
        (array Dtypes.const) (List.hd reduction) v)
    devices;
  equal ~msg:"devices" int (List.length reduction) (List.length devices)

let reduces ?(settings = []) red = holds_reduction red (handled ~settings red)
let computes ?(settings = []) red = holds_reduction red (created ~settings red)

let concrete name =
  List.for_all
    (function Ops.Int _ -> true | Sym _ -> false)
    (Ops.shape (allreduce name))

(* An allreduce of 300,000 elements takes a second. *)
let case name =
  match Ops.numel (allreduce name) with
  | Int n when n > 100_000 -> slow
  | _ -> test

let recorded_values =
  let concrete = List.filter (fun (name, _) -> concrete name) cases in
  group "handle_allreduce › values"
    (List.map
       (fun (name, settings) ->
         case name (name ^ " reduces its shards") (fun () ->
             reduces ~settings (allreduce name)))
       concrete
    @ List.map
        (fun (name, settings) ->
          case name (name ^ "'s function reduces its shards") (fun () ->
              computes ~settings (allreduce name)))
        concrete)

(* Generated allreduces: two to five devices, shapes of one or two axes, and
   every setting of the algorithms. *)

type drawn = {
  n : int;
  shape : int list;
  op : Op.t;
  one_device : bool;
  settings : int * int * int;
}

let pp_drawn ppf d =
  let r, a, h = d.settings in
  Format.fprintf ppf "%s of [%s] on %d devices%s, RING=%d ALL2ALL=%d NODES=%d"
    (Op.name d.op)
    (String.concat "; " (List.map string_of_int d.shape))
    d.n
    (if d.one_device then " to one" else "")
    r a h

let drawn =
  Gen.(
    let+ n = int_range 2 5
    and+ shape = list ~size:(int_range 1 2) (int_range 1 6)
    and+ op = of_list [ Op.Add; Max ]
    and+ one_device = bool
    and+ settings = triple (int_range 0 2) (int_range 0 2) (int_range 0 3) in
    { n; shape; op; one_device; settings })
  |> Gen.with_pp pp_drawn

let red_of d =
  let names = List.init d.n (Printf.sprintf "CPU:%d") in
  let size = List.fold_left ( * ) 1 d.shape in
  let buf = Ops.new_buffer (Multi names) size Float32 in
  let target = if d.one_device then Ops.Single "CPU:0" else Multi names in
  Ops.allreduce
    (Ops.reshape buf (List.map (fun n -> Ops.Int n) d.shape))
    d.op target

let bindings d =
  let r, a, h = d.settings in
  [ ring r; all2all a; nodes h ]

let generated =
  group "handle_allreduce › laws"
    [
      prop "an expansion reduces its shards" drawn (fun d ->
          let numel = List.fold_left ( * ) 1 d.shape in
          let _, _, h = d.settings in
          cover "a chunk is empty" (numel < d.n);
          cover "devices form nodes" (h > 0 && d.n mod h = 0);
          reduces ~settings:(bindings d) (red_of d));
      prop "a function reduces its shards" drawn (fun d ->
          computes ~settings:(bindings d) (red_of d));
    ]

(* Algorithms

   The algorithm an allreduce takes, as its printout names it: all-to-all when
   forced, or past the threshold on more than two devices when allowed; else the
   ring, under the same conditions; else the naive one. A symbolic shape takes
   the naive one. *)

type choice = { devices : int; numel : int; ring : int; all2all : int }

let label c =
  let many = c.devices > 2 && c.numel > 256_000 in
  let a2a = c.all2all >= 2 || (many && c.all2all >= 1) in
  if a2a then "ALL2ALL"
  else if c.ring >= 2 || (many && c.ring >= 1) then "RING"
  else "NAIVE"

let choice =
  Gen.(
    let+ devices = int_range 2 5
    and+ numel = of_list [ 1; 256_000; 256_001 ]
    and+ ring = int_range 0 2
    and+ all2all = int_range 0 2 in
    { devices; numel; ring; all2all })
  |> Gen.with_pp (fun ppf c ->
      Format.fprintf ppf "%d elements on %d devices, RING=%d ALL2ALL=%d" c.numel
        c.devices c.ring c.all2all)

let chosen c =
  let d =
    {
      n = c.devices;
      shape = [ c.numel ];
      op = Add;
      one_device = false;
      settings = (c.ring, c.all2all, 0);
    }
  in
  ignore
    (handled ~settings:(Setting.B (Setting.debug, 2) :: bindings d) (red_of d));
  chomp (output ())

(* The devices a copy moves a value between, one pair per target. *)
let route c =
  let names = function Ops.Single d -> [ d ] | Multi l -> l in
  let from =
    match Ops.device (Ops.nth c 0) with
    | Some (Single d) -> d
    | _ -> fail "a copy's source is on one device"
  in
  match Ops.arg c with
  | Device d -> List.map (fun t -> (from, t)) (names d)
  | _ -> fail "a copy has a target"

let crossings u =
  List.concat_map route
    (List.filter (fun n -> Ops.op n = Copy) (Ops.toposort ~calls:Enter u))
  |> List.filter (fun (a, b) -> a <> b)

let four = List.init 4 (Printf.sprintf "CPU:%d")
let next k = (List.nth four k, List.nth four ((k + 1) mod 4))

let four_of settings =
  handled ~settings
    (red_of
       {
         n = 4;
         shape = [ 400 ];
         op = Add;
         one_device = false;
         settings = (0, 0, 0);
       })

let algorithms =
  let route = pair string string in
  group "handle_allreduce › algorithms"
    [
      prop "the algorithm is the first that applies" choice (fun c ->
          equal string
            (Printf.sprintf "%s ALLREDUCE %dx%d | dtypes.float" (label c)
               c.devices c.numel)
            (chosen c));
      test "a ring moves values only to the next device" (fun () ->
          let moves = crossings (four_of [ ring 2 ]) in
          equal int 24 (List.length moves);
          equal (slist route compare) (List.init 4 next)
            (List.sort_uniq compare moves));
      test "all-to-all moves values between every two devices" (fun () ->
          let moves = crossings (four_of [ all2all 2 ]) in
          equal int 24 (List.length moves);
          equal int 12 (List.length (List.sort_uniq compare moves)));
      test "a naive allreduce copies each shard to every device" (fun () ->
          equal int 12 (List.length (crossings (four_of [ ring 0 ]))));
      test "a symbolic shape is kept, and a function stores its greatest"
        (fun () ->
          let red = allreduce "naive_symbolic" in
          let settings = settings "naive_symbolic" in
          let sizes u = List.map Ops.sint_to_uop (Ops.shape u) in
          equal (list uop) (sizes red) (sizes (handled ~settings red));
          equal (list uop) (sizes red) (sizes (created ~settings red));
          equal (list int) [ 64 ]
            (List.map Ops.max_numel (made_storage red (created ~settings red))));
    ]

(* Rules *)

let on_one_device = Ops.new_buffer (Single "CPU") 8 Float32

let rules =
  let red =
    Ops.v Allreduce ~src:[ on_one_device ]
      ~arg:(Allreduce { op = Add; device = Single "CPU" })
  in
  group "rules"
    [
      test "a hierarchical allreduce to one device lands there" (fun () ->
          let red =
            red_of
              {
                n = 2;
                shape = [ 1 ];
                op = Add;
                one_device = true;
                settings = (0, 0, 1);
              }
          in
          let settings = [ nodes 1 ] in
          reduces ~settings red;
          computes ~settings red);
      test "a value on one device is left as it is" (fun () ->
          is_none (Allreduce.handle_allreduce red));
      test "a function of a value on one device is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"several devices") (fun () ->
              Allreduce.create_allreduce_function red));
      test "a function of a node that is not an allreduce is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"not an allreduce")
            (fun () -> Allreduce.create_allreduce_function on_one_device));
      test "a function is named allreduce and compiled on its own" (fun () ->
          let calls =
            List.filter
              (fun n -> Ops.op n = Call)
              (Ops.toposort ~calls:Enter
                 (created (allreduce "naive_two_devices")))
          in
          equal
            (list (pair (option string) bool))
            [ (Some "allreduce", true) ]
            (List.map
               (fun c ->
                 match Ops.arg c with
                 | Call { name; precompile; _ } -> (name, precompile)
                 | _ -> (None, false))
               calls));
    ]

let () =
  exit
    (run "Tolk.Allreduce"
       [
         expansions;
         functions;
         printouts;
         recorded_values;
         generated;
         algorithms;
         rules;
       ])
