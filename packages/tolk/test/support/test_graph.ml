open Windtrap
open Tolk

let node = Testable.make ~pp:Ops.pp ~equal:Ops.equal
let int n = Ops.v ~arg:(Const (`Int (Bigint.of_int n))) Const
let cast dt u = Ops.v ~src:[ u ] ~arg:(Dtype dt) Cast
let x = cast Float32 (int 1)
let half = cast Float16 (int 1)

let n =
  Ops.v
    ~arg:
      (Param
         (Ops.param_arg ~slot:(-1)
            ~vmin_vmax:(`Int Bigint.zero, `Int (Bigint.of_int 9))
            ~name:"n" ~addrspace:Alu Weak_int))
    Param

let buffer =
  Ops.v
    ~src:[ Ops.size_src (Some 16) ]
    ~arg:(Param (Ops.param_arg ~slot:0 ~device:"CPU" Float32))
    Buffer

(* The laws *)

let goldens =
  group "a golden reads into the graph that writes it"
    (List.map
       (fun file -> Golden.graph file (fun () -> Golden.sink file))
       [ "matmul.golden"; "sum_of_cast.golden"; "sum_of_cast_kernels.golden" ])

(* One node for each kind of argument. *)
let kinds =
  let k = Ops.kernel_info in
  [
    ("an integer", int 3);
    ( "an integer beyond 64 bits",
      Ops.v ~arg:(Const (`Int (Bigint.shift_left Bigint.one 200))) Const );
    ("a boolean", Ops.v ~arg:(Const (`Bool true)) Const);
    ("a negative zero", Ops.v ~arg:(Const (`Float (-0.0))) Const);
    ("a NaN", Ops.v ~arg:(Const (`Float Float.nan)) Const);
    ("Invalid", Ops.v ~arg:(Const `Invalid) Const);
    ("a cast", x);
    ("a variable", n);
    ("a buffer", buffer);
    ( "a buffer past a 16-byte boundary",
      Ops.v
        ~src:[ Ops.size_src (Some 16) ]
        ~arg:
          (Param
             (Ops.param_arg ~slot:0 ~device:"CPU" ~phase:8 Float32))
        Buffer );
    ( "a scalar parameter",
      Ops.v
        ~src:[ Ops.size_src None ]
        ~arg:
          (Param
             (Ops.param_arg ~slot:1 ~addrspace:Alu ~volatile:true
                ~bind_on_realize:true
                ~bound:(`Int (Bigint.of_int 4))
                Int32))
        Param );
    ( "a range",
      Ops.v
        ~src:[ int 16 ]
        ~arg:(Range { axis_type = Weak; axis_id = [ 1; 0 ] })
        Range );
    ( "a reduction",
      Ops.v ~src:[ x ] ~arg:(Reduce { op = Add; num_axes = 1 }) Reduce );
    ("a copy", Ops.v ~src:[ x ] ~arg:(Device "CPU") Copy);
    ("a launch dimension", Ops.v ~src:[ int 4 ] ~arg:(String "gidx0") Special);
    ( "a flagged load",
      Ops.v ~src:[ Ops.index buffer [ int 0 ] ] ~arg:(String "nontemporal") Load
    );
    ( "a source with every escape",
      Ops.v ~arg:(String "a\"b\\c\nd\te\x00\xc3\xa9") Source );
    ("bytes", Ops.v ~arg:(Bytes "\x7fELF\x00") Binary);
    ("a named linear", Ops.v ~src:[ x ] ~arg:(String "kernargs") Linear);
    ( "custom code",
      Ops.call
        (Ops.custom_function ~dtype:Float32 ~fmt:true "{0}+1" [])
        [ x ] );
    ( "a stage",
      Ops.v ~src:[ buffer ]
        ~arg:
          (Bufferize
             {
               device = Some "CPU";
               addrspace = Local;
               keep = Whole;
             })
        Stage );
    ( "a kernel with every optimisation and symbolic estimates",
      Ops.sink
        ~kernel:
          (k ~name:"r_16"
             ~applied_opts:
               [
                 Tc { axis = 0; tc_select = -1; tc_opt = 2; use_tc = 1 };
                 Split { axis = 1; amount = 4; target = Upcast; top = true };
                 Padto { axis = 0; amount = 32 };
                 Swap { axis = 0; with_axis = 1 };
               ]
             ~opts_to_apply:[]
             ~estimates:{ ops = Sym n; lds = Int 4; mem = Int 0 }
             ~beam:2 ())
        [ x ] );
    ( "a program",
      Ops.v
        ~arg:
          (Program
             {
               global_size = [ Sym n; Int 1; Int 1 ];
               local_size = [ Int 1; Int 1; Int 1 ];
               params = [ n ];
               globals = [ 0 ];
               outs = [ 0 ];
               ins = [];
               target =
                 {
                   device = "CPU";
                   renderer = "CLANG";
                   arch = "";
                   interface = "";
                   indices = "";
                 };
             })
        Program );
    ( "a call",
      Ops.v
        ~src:[ Ops.sink [ x ]; buffer ]
        ~arg:
          (Call { name = Some "f"; precompile = true; inline = false })
        Call );
    ( "a tensor core product",
      Ops.v ~src:[ half; half; x ]
        ~arg:
          (Wmma
             {
               dims = (8, 16, 16);
               dtype_in = Float16;
               threads = 32;
               upcast_axes =
                 Some
                   ( [ ((Upcast, [ 0 ]), 2) ],
                     [ ((Upcast, [ 1 ]), 2) ],
                     [ ((Upcast, [ 2 ]), 2) ] );
             })
        Wmma );
    ( "a chain of 100000 nodes",
      List.fold_left
        (fun u _ -> Ops.v ~src:[ u ] Neg)
        x (List.init 100_000 Fun.id) );
  ]

let round_trip =
  cases "a graph reads back as itself" ~id:fst kinds (fun (_, u) ->
      equal node u (Graph.of_string (Graph.to_string u)))

(* The text *)

let text =
  group "text"
    [
      test "writes one line per node, sources first" (fun () ->
          equal Windtrap.text
            "0 Ops.CONST dtypes.weakint [] 1\n\
             1 Ops.CAST dtypes.f32 [0] dtypes.f32\n\
             2 Ops.ADD dtypes.f32 [1, 1]\n"
            (Graph.to_string (Ops.v ~src:[ x; x ] Add)));
      test "writes a record's fields that differ from their default" (fun () ->
          equal Windtrap.text
            "0 Ops.CONST dtypes.weakint [] 16\n\
             1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, \
             device=\"CPU\")\n"
            (Graph.to_string buffer));
      test "writes an argument's nodes before the node that holds them"
        (fun () ->
          let program = List.assoc "a program" kinds in
          equal Windtrap.text
            "0 Ops.PARAM dtypes.weakint [] ParamArg(slot=-1, \
             dtype=dtypes.weakint, vmin_vmax=(0, 9), name=\"n\", \
             addrspace=AddrSpace.ALU)\n\
             1 Ops.PROGRAM dtypes.void [] ProgramInfo(global_size=(%0, 1, 1), \
             params=(%0,), globals=(0,), outs=(0,), \
             target=Target(device=\"CPU\", renderer=\"CLANG\"))\n"
            (Graph.to_string program));
    ]

(* The errors *)

let rejects substring text =
  raises_match (Exn.failure ~sub:substring) (fun () -> Graph.of_string text)

(* A graph whose slot and range axes lie past every number handed out so far, at
   the drawn distances: reading it back takes them all. *)
let numbers =
  prop "a fresh number names no slot or range of a graph read back"
    Gen.(
      pair (int_range 1 1000) (list ~size:(int_range 1 3) (int_range 1 1000)))
    (fun (slot, axes) ->
      let next = Ops.unique_num () + 1 in
      let axis_id = List.map (( + ) next) axes in
      let buffer =
        Ops.v
          ~src:[ Ops.size_src (Some 4) ]
          ~arg:
            (Param
               (Ops.param_arg ~slot:(next + slot) ~device:"CPU" Float32))
          Buffer
      in
      let r = Ops.range ~axis_type:Loop (Int 3) axis_id in
      let text = Graph.to_string (Ops.sink [ Ops.index buffer [ r ] ]) in
      ignore (Graph.of_string text);
      greater Windtrap.int
        ~than:(List.fold_left max (next + slot) axis_id)
        (Ops.unique_num ()))


(* Damage *)

(* Damage to a graph's text or to its entry on disk, at a place given as a
   fraction of the text's length or of the number of places of its kind: a
   byte flipped, the text cut short, one of its numbers or one of its slots
   replaced by a huge number, such as one past every integer. *)
type damage =
  | Flip of float * int
  | Cut of float
  | Huge_number of float * string
  | Huge_slot of float * string

let pp_damage ppf = function
  | Flip (at, bits) -> Format.fprintf ppf "Flip (%g, 0x%02x)" at bits
  | Cut at -> Format.fprintf ppf "Cut %g" at
  | Huge_number (at, n) -> Format.fprintf ppf "Huge_number (%g, %s)" at n
  | Huge_slot (at, n) -> Format.fprintf ppf "Huge_slot (%g, %s)" at n

let huge =
  [
    string_of_int max_int;
    string_of_int (1 lsl 40);
    "1000000000000000000";
    "18446744073709551616";
    "1000000000000000000000000000000";
  ]

let damage =
  let open Gen in
  let at = float_range 0. 1. in
  with_pp pp_damage
    (one_of
       [
         map (fun (a, b) -> Flip (a, b)) (pair at (int_range 1 255));
         map (fun a -> Cut a) at;
         map (fun (a, n) -> Huge_number (a, n)) (pair at (of_list huge));
         map (fun (a, n) -> Huge_slot (a, n)) (pair at (of_list huge));
       ])

let nth_of n at = min (n - 1) (int_of_float (at *. float_of_int n))

(* The places [(start, stop)] of the matches of [ok] in [s]: its runs of
   characters [ok] holds for. *)
let runs ok s =
  let n = String.length s in
  let rec from i acc =
    if i >= n then List.rev acc
    else if not (ok s.[i]) then from (i + 1) acc
    else
      let j = ref i in
      while !j < n && ok s.[!j] do incr j done;
      from !j ((i, !j) :: acc)
  in
  from 0 []

let replace s (start, stop) by =
  String.sub s 0 start ^ by ^ String.sub s stop (String.length s - stop)

(* The numbers written after "slot=" in [s]. *)
let slots s =
  List.filter
    (fun (start, _) -> start >= 5 && String.sub s (start - 5) 5 = "slot=")
    (runs (function '0' .. '9' | '-' -> true | _ -> false) s)

(* [s] with [d] done to it; [None] if [s] has no place of [d]'s kind. *)
let damaged d s =
  let n = String.length s in
  let place places at by =
    match places with
    | [] -> None
    | ps -> Some (replace s (List.nth ps (nth_of (List.length ps) at)) by)
  in
  match d with
  | Flip (at, bits) ->
      let i = nth_of n at in
      Some
        (replace s (i, i + 1)
           (String.make 1 (Char.chr (Char.code s.[i] lxor bits))))
  | Cut at -> Some (String.sub s 0 (nth_of n at))
  | Huge_number (at, by) ->
      place (runs (function '0' .. '9' -> true | _ -> false) s) at by
  | Huge_slot (at, by) -> place (slots s) at by

let classify_damage = function
  | Flip _ -> cover "a byte flipped" true
  | Cut _ -> cover "cut short" true
  | Huge_number _ -> cover "a huge number" true
  | Huge_slot _ -> cover "a huge slot" true

(* The graphs damaged: every kind but the long chain, and a buffer indexed
   by a range, whose slot and axis are numbers the reading takes. *)
let damageable =
  let r = Ops.range ~axis_type:Loop (Int 3) [ 2; 1 ] in
  ("an indexed buffer", Ops.sink [ Ops.index buffer [ r ] ])
  :: List.filter (fun (k, _) -> k <> "a chain of 100000 nodes") kinds

(* A damaged graph, drawn with the damage: a slot is damaged in a graph that
   has one. *)
let case =
  let open Gen in
  let named = with_pp (fun ppf (k, _) -> Format.pp_print_string ppf k) in
  let slotted =
    List.filter (fun (_, u) -> slots (Graph.to_string u) <> []) damageable
  in
  let* d = damage in
  let+ g =
    named
      (of_list (match d with Huge_slot _ -> slotted | _ -> damageable))
  in
  (g, d)

(* Reading damaged text fails, or reads a graph, without hanging: a huge slot
   is taken at once, and one past every number handed out leaves none. *)
let damaged_text =
  prop ~timeout:20. "damaged text reads or fails, at once"
    case
    (fun ((_, u), d) ->
      match damaged d (Graph.to_string u) with
      | None -> reject ()
      | Some text -> (
          classify_damage d;
          match Graph.of_string text with
          | _ -> (
              match d with
              | Huge_slot (_, n) -> (
                  match int_of_string_opt n with
                  | Some n -> greater Windtrap.int ~than:n (Ops.unique_num ())
                  | None -> ())
              | _ -> ())
          | exception Failure _ -> ()))

(* A graph whose entry on disk is damaged reads back as itself, or is made
   anew. The lengths of an entry damaged in its value's numbers are stated
   anew, so that they do not show it. *)
let damaged_entry =
  let count = ref 0 in
  prop ~timeout:20. "a damaged entry reads back as its graph, or misses"
    case
    (fun ((_, u), d) ->
      if Sys.getenv_opt "CACHEDB" = None then
        skip ~reason:"CACHEDB is unset" ();
      incr count;
      let table = Printf.sprintf "test_graph_damage_%d" !count in
      let cached make =
        Graph.cached ~table ~key:"k" ~valid:(fun _ -> true) make
      in
      ignore (cached (fun () -> u));
      let entry = List.hd (Disk_cache.entries ~table Helpers.cachedb) in
      let e = In_channel.with_open_bin entry In_channel.input_all in
      let or_reject = function Some s -> s | None -> reject () in
      let e' =
        match d with
        | Flip _ | Cut _ -> or_reject (damaged d e)
        | Huge_number _ | Huge_slot _ ->
            Disk_cache.restated (fun v -> or_reject (damaged d v)) e
      in
      classify_damage d;
      Out_channel.with_open_bin entry (fun oc -> output_string oc e');
      let made = Ops.sink [] in
      match cached (fun () -> made) with
      | g, true -> equal node ~msg:"read back" u g
      | g, false -> equal node ~msg:"made anew" made g)

(* A graph whose entry the process may not read, as in a cache another user
   shares, is made anew. *)
let unreadable_entry =
  test "an entry that cannot be read misses" (fun () ->
      if Sys.getenv_opt "CACHEDB" = None then
        skip ~reason:"CACHEDB is unset" ();
      if Sys.win32 then skip ~reason:"no file mode forbids reading" ();
      if Unix.geteuid () = 0 then skip ~reason:"root reads any file" ();
      let table = "test_graph_unreadable" in
      let cached make =
        Graph.cached ~table ~key:"k" ~valid:(fun _ -> true) make
      in
      ignore (cached (fun () -> x));
      List.iter
        (fun e -> Unix.chmod e 0)
        (Disk_cache.entries ~table Helpers.cachedb);
      let made = Ops.sink [] in
      match cached (fun () -> made) with
      | g, read -> equal (pair node bool) (made, false) (g, read))

let errors =
  group "reading"
    [
      test "rejects an empty text" (fun () -> rejects "no node" "");
      test "rejects a written dtype that the node does not derive" (fun () ->
          rejects
            "node 1: the node's data type is dtypes.f32, not the written \
             dtypes.i32"
            "0 Ops.CONST dtypes.weakint [] 1\n\
             1 Ops.CAST dtypes.i32 [0] dtypes.f32\n");
      test "rejects a source that is not an earlier line" (fun () ->
          rejects "node 0: node 0 is not an earlier line"
            "0 Ops.NEG dtypes.weakint [0]\n");
      test "rejects an index that is not the line's position" (fun () ->
          rejects "node 0: index 1" "1 Ops.CONST dtypes.weakint [] 1\n");
      test "rejects an unknown operation" (fun () ->
          rejects "node 0: " "0 Ops.FOO dtypes.weakint [] 1\n");
      test "rejects an argument on an operation that takes none" (fun () ->
          rejects "Ops.NOOP takes no argument" "0 Ops.NOOP dtypes.void [] 1\n");
      test "rejects a split into an axis type no split makes" (fun () ->
          rejects "a split cannot make an axis of type AxisType.WEAK"
            "0 Ops.SINK dtypes.void [] \
             KernelInfo(applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, \
             AxisType.WEAK)),))\n");
      test "rejects an unknown field" (fun () ->
          rejects "KernelInfo has no field size"
            "0 Ops.SINK dtypes.void [] KernelInfo(size=1)\n");
      test "rejects text after the argument" (fun () ->
          rejects "unexpected text at column 32"
            "0 Ops.CONST dtypes.weakint [] 1 2\n");
    ]

let () =
  exit
    (run "Graph"
       [
         goldens;
         round_trip;
         text;
         numbers;
         damaged_text;
         damaged_entry;
         unreadable_entry;
         errors;
       ])
