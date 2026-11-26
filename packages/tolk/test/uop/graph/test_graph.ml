open Windtrap
open Tolk

let node = Testable.make ~pp:Ops.pp ~equal:Ops.equal
let int n = Ops.v ~arg:(Const (`Int (Bigint.of_int n))) Const
let const c = Ops.v ~arg:(Const c) Const
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

let program =
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
    Program

let kernel ?opts_to_apply applied_opts =
  Ops.sink ~kernel:(Ops.kernel_info ~applied_opts ?opts_to_apply ()) [ x ]

(* to_string *)

(* The argument [u]'s line writes: what follows its sources. *)
let written_arg u =
  let text = Graph.to_string u in
  let lines = String.split_on_char '\n' (String.trim text) in
  let line = List.nth lines (List.length lines - 1) in
  match String.index_opt line ']' with
  | Some i when i + 2 <= String.length line ->
      String.sub line (i + 2) (String.length line - i - 2)
  | _ -> ""

(* Each value as the format's table in test/README.md writes it. *)
let values =
  [
    ("an integer", int (-3), "-3");
    ( "an integer past 64 bits",
      const (`Int (Bigint.shift_left Bigint.one 128)),
      "340282366920938463463374607431768211456" );
    ("a float", const (`Float 1.0), "1.0");
    ("negative zero", const (`Float (-0.0)), "-0.0");
    ("a float below 1e-4", const (`Float 1e-5), "1e-05");
    ("a float of 1e16", const (`Float 1e16), "1e+16");
    ("infinity", const (`Float Float.infinity), "inf");
    ("negative infinity", const (`Float Float.neg_infinity), "-inf");
    ("the positive quiet NaN", const (`Float Args.quiet), "nan");
    ( "a NaN of other bits",
      const (`Float (Int64.float_of_bits 0xfffffc0000000000L)),
      "nan(0xfffffc0000000000)" );
    ("the invalid constant", const `Invalid, "Invalid");
    ("true", const (`Bool true), "True");
    ("false", const (`Bool false), "False");
    ( "a string of every escape",
      Ops.v ~arg:(String "a\"b\\c\nd\te\x00\x7f\xc3\xa9") Source,
      {|"a\"b\\c\nd\te\x00\x7f\xc3\xa9"|} );
    ("bytes", Ops.v ~arg:(Bytes "\x7fELF") Binary, {|b"\x7fELF"|});
    ("a data type", half, "dtypes.f16");
    ( "an enumeration",
      Ops.v ~src:[ x ] ~arg:(Reduce { op = Add; num_axes = 1 }) Reduce,
      "(Ops.ADD, 1)" );
    ( "a range's axis type and id",
      Ops.v
        ~src:[ int 16 ]
        ~arg:(Range { axis_type = Weak; axis_id = [ 1; 0 ] })
        Range,
      "(AxisType.WEAK, 1, 0)" );
    ( "a tuple of one",
      kernel [ Padto { axis = 0; amount = 32 } ],
      "KernelInfo(applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32),))" );
    ( "an empty tuple",
      kernel ~opts_to_apply:[] [],
      "KernelInfo(opts_to_apply=())" );
  ]

let to_string =
  group "to_string"
    [
      (* The format's example: index, op, dtype, sources, argument. *)
      test "writes one line per node, sources first" (fun () ->
          equal Windtrap.text
            "0 Ops.CONST dtypes.weakint [] 1\n\
             1 Ops.CAST dtypes.f32 [0] dtypes.f32\n\
             2 Ops.ADD dtypes.f32 [1, 1]\n"
            (Graph.to_string (Ops.v ~src:[ x; x ] Add)));
      (* "A node comes after its sources and after the nodes that its arg
         holds", written as [%] and its index. *)
      test "writes an argument's nodes before the node that holds them"
        (fun () ->
          equal Windtrap.text
            "0 Ops.PARAM dtypes.weakint [] ParamArg(slot=-1, \
             dtype=dtypes.weakint, vmin_vmax=(0, 9), name=\"n\", \
             addrspace=AddrSpace.ALU)\n\
             1 Ops.PROGRAM dtypes.void [] ProgramInfo(global_size=(%0, 1, 1), \
             params=(%0,), globals=(0,), outs=(0,), \
             target=Target(device=\"CPU\", renderer=\"CLANG\"))\n"
            (Graph.to_string program));
      (* "fields in declaration order, each field left out when it holds its
         default" *)
      test "writes a record's fields that differ from their default" (fun () ->
          equal Windtrap.text
            "0 Ops.CONST dtypes.weakint [] 16\n\
             1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, \
             device=\"CPU\")\n"
            (Graph.to_string buffer));
      cases "writes each value as the format's table states"
        ~id:(fun (name, _, _) -> name)
        values
        (fun (_, u, text) -> equal Windtrap.string text (written_arg u));
    ]

(* of_string *)

(* One node for each kind of argument. *)
let kinds =
  let k = Ops.kernel_info in
  [
    int 3;
    const (`Int (Bigint.shift_left Bigint.one 200));
    const (`Bool true);
    const (`Float (-0.0));
    const (`Float Float.nan);
    const `Invalid;
    x;
    n;
    buffer;
    Ops.v
      ~src:[ Ops.size_src (Some 16) ]
      ~arg:(Param (Ops.param_arg ~slot:0 ~device:"CPU" ~phase:8 Float32))
      Buffer;
    Ops.v
      ~src:[ Ops.size_src None ]
      ~arg:
        (Param
           (Ops.param_arg ~slot:1 ~addrspace:Alu ~volatile:true
              ~bind_on_realize:true
              ~bound:(`Int (Bigint.of_int 4))
              Int32))
      Param;
    Ops.v
      ~src:[ int 16 ]
      ~arg:(Range { axis_type = Weak; axis_id = [ 1; 0 ] })
      Range;
    Ops.v ~src:[ x ] ~arg:(Reduce { op = Add; num_axes = 1 }) Reduce;
    Ops.v ~src:[ x ] ~arg:(Device "CPU") Copy;
    Ops.v ~src:[ int 4 ] ~arg:(String "gidx0") Special;
    Ops.v ~src:[ Ops.index buffer [ int 0 ] ] ~arg:(String "nontemporal") Load;
    Ops.v ~arg:(String "a\"b\\c\nd\te\x00\xc3\xa9") Source;
    Ops.v ~arg:(Bytes "\x7fELF\x00") Binary;
    Ops.v ~src:[ x ] ~arg:(String "kernargs") Linear;
    Ops.call (Ops.custom_function ~dtype:Float32 ~fmt:true "{0}+1" []) [ x ];
    Ops.v ~src:[ buffer ]
      ~arg:(Bufferize { device = Some "CPU"; addrspace = Local; keep = Whole })
      Stage;
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
      [ x ];
    program;
    Ops.v
      ~src:[ Ops.sink [ x ]; buffer ]
      ~arg:
        (Call
           {
             name = Some "f";
             precompile = true;
             inline = false;
             loop = false;
             write = [];
           })
      Call;
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
      Wmma;
    (* The two rare covers below. *)
    Ops.v ~src:[ buffer ]
      ~arg:
        (Bufferize { device = Some "CPU"; addrspace = Global; keep = Argument })
      Stage;
    Ops.v
      ~src:[ Ops.size_src None ]
      ~arg:
        (Param
           (Ops.param_arg
              ~vmin_vmax:(`Float 0., `Float Args.signalling)
              ~bound:(`Float Args.signalling) ~addrspace:Alu ~slot:1 Float32))
      Param;
  ]

let holds_nan = function `Float x -> Float.is_nan x | _ -> false

(* A schedule kept on disk is its graph's text: the text reads back as the
   graph, so two graphs of one text are one graph. *)
let reads_back u =
  cover "a stage of a custom kernel's argument"
    (match Ops.arg u with
    | Bufferize { keep = Argument; _ } -> true
    | _ -> false);
  cover "a parameter holding a NaN"
    (match Ops.arg u with
    | Param { vmin_vmax; bound; _ } ->
        Option.fold ~none:false
          ~some:(fun (a, b) -> holds_nan a || holds_nan b)
          vmin_vmax
        || Option.fold ~none:false ~some:holds_nan bound
    | _ -> false);
  Law.round_trip node Windtrap.string Graph.to_string Graph.of_string u

(* A graph whose slot and range axes lie past every number handed out so far, at
   the drawn distances: reading it back takes them all. *)
let takes_numbers (slot, axes) =
  let next = Ops.unique_num () + 1 in
  let axis_id = List.map (( + ) next) axes in
  let buffer =
    Ops.v
      ~src:[ Ops.size_src (Some 4) ]
      ~arg:(Param (Ops.param_arg ~slot:(next + slot) ~device:"CPU" Float32))
      Buffer
  in
  let r = Ops.range ~axis_type:Loop (Int 3) axis_id in
  ignore
    (Graph.of_string (Graph.to_string (Ops.sink [ Ops.index buffer [ r ] ])));
  greater Windtrap.int
    ~than:(List.fold_left max (next + slot) axis_id)
    (Ops.unique_num ())

(* Texts that do not read, each with the line it names and a word of what is
   wrong. *)
let faults =
  [
    ( "a line of an unknown operation",
      0,
      "FOO",
      "0 Ops.FOO dtypes.weakint [] 1\n" );
    ( "a line with text after its argument",
      0,
      "unexpected text",
      "0 Ops.CONST dtypes.weakint [] 1 2\n" );
    ( "an index that is not its line's position",
      0,
      "index 1",
      "1 Ops.CONST dtypes.weakint [] 1\n" );
    ( "a source that is not an earlier line",
      0,
      "node 0 is not an earlier line",
      "0 Ops.NEG dtypes.weakint [0]\n" );
    ( "a node in an argument that is not an earlier line",
      0,
      "node 1 is not an earlier line",
      "0 Ops.PROGRAM dtypes.void [] ProgramInfo(params=(%1,))\n\
       1 Ops.CONST dtypes.weakint [] 1\n" );
    ( "an argument on an operation that takes none",
      0,
      "Ops.NOOP takes no argument",
      "0 Ops.NOOP dtypes.void [] 1\n" );
    ( "a split into an axis type no split makes",
      0,
      "a split cannot make an axis of type AxisType.WEAK",
      "0 Ops.SINK dtypes.void [] KernelInfo(applied_opts=(Opt(op=OptOps.SPLIT, \
       axis=0, arg=(2, AxisType.WEAK)),))\n" );
    ( "an unknown field",
      0,
      "KernelInfo has no field size",
      "0 Ops.SINK dtypes.void [] KernelInfo(size=1)\n" );
    ( "a call that precompiles its backward pass",
      1,
      "backward",
      "0 Ops.SINK dtypes.void []\n\
       1 Ops.CALL dtypes.void [0] CallInfo(precompile_backward=True)\n" );
    ( "a written dtype that the node does not derive",
      1,
      "the node's data type is dtypes.f32, not the written dtypes.i32",
      "0 Ops.CONST dtypes.weakint [] 1\n1 Ops.CAST dtypes.i32 [0] dtypes.f32\n"
    );
    ( "a slot of max_int",
      1,
      "leaves no number",
      Printf.sprintf
        "0 Ops.CONST dtypes.weakint [] 16\n\
         1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=%d, dtype=dtypes.f32, \
         device=\"CPU\")\n"
        max_int );
    ( "a range axis of max_int",
      1,
      "leaves no number",
      Printf.sprintf
        "0 Ops.CONST dtypes.weakint [] 3\n\
         1 Ops.RANGE dtypes.weakint [0] (AxisType.LOOP, %d)\n"
        max_int );
  ]

let names_node line fault = function
  | Failure m ->
      String.starts_with ~prefix:(Printf.sprintf "node %d: " line) m
      && Exn.failure ~sub:fault (Failure m)
  | _ -> false

(* Damage *)

(* Damage to a graph's text or to its entry on disk, at a place given as a
   fraction of the text's length or of the number of places of its kind: a byte
   flipped, the text cut short, one of its numbers or one of its slots replaced
   by a huge number, such as one past every integer. *)
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
      while !j < n && ok s.[!j] do
        incr j
      done;
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
        (replace s
           (i, i + 1)
           (String.make 1 (Char.chr (Char.code s.[i] lxor bits))))
  | Cut at -> Some (String.sub s 0 (nth_of n at))
  | Huge_number (at, by) ->
      place (runs (function '0' .. '9' -> true | _ -> false) s) at by
  | Huge_slot (at, by) -> place (slots s) at by

let classify_damage d =
  cover "a byte flipped" (match d with Flip _ -> true | _ -> false);
  cover "cut short" (match d with Cut _ -> true | _ -> false);
  cover "a huge number" (match d with Huge_number _ -> true | _ -> false);
  cover "a huge slot" (match d with Huge_slot _ -> true | _ -> false)

(* The graphs damaged: every kind, and a buffer indexed by a range, whose slot
   and axis are numbers the reading takes. *)
let damageable =
  let r = Ops.range ~axis_type:Loop (Int 3) [ 2; 1 ] in
  Ops.sink [ Ops.index buffer [ r ] ] :: kinds

(* A damaged graph, drawn with the damage: a slot is damaged in a graph that has
   one. *)
let case =
  let open Gen in
  let shown =
    with_pp (fun ppf u -> Format.pp_print_string ppf (Graph.to_string u))
  in
  let slotted =
    List.filter (fun u -> slots (Graph.to_string u) <> []) damageable
  in
  let* d = damage in
  let+ g =
    shown (of_list (match d with Huge_slot _ -> slotted | _ -> damageable))
  in
  (g, d)

(* Reading damaged text fails, or reads a graph; a huge slot that reads is
   taken, and one past every number handed out leaves none. *)
let reads_damaged (u, d) =
  classify_damage d;
  match damaged d (Graph.to_string u) with
  | None -> reject ()
  | Some text -> (
      match Graph.of_string text with
      | exception Failure _ -> ()
      | _ -> (
          match d with
          | Huge_slot (_, n) -> (
              match int_of_string_opt n with
              | Some n -> greater Windtrap.int ~than:n (Ops.unique_num ())
              | None -> ())
          | _ -> ()))

let of_string =
  group "of_string"
    [
      prop ~count:2000 "a graph of any argument reads back as itself"
        ~examples:kinds Args.node reads_back;
      test "a graph of any depth reads back as itself" (fun () ->
          let chain =
            List.fold_left
              (fun u _ -> Ops.v ~src:[ u ] Neg)
              x (List.init 100_000 Fun.id)
          in
          equal node chain (Graph.of_string (Graph.to_string chain)));
      group "reads each golden into the graph that writes it"
        [
          Golden.graph "matmul.golden" (fun () -> Golden.sink "matmul.golden");
          Golden.graph "sum_of_cast.golden" (fun () ->
              Golden.sink "sum_of_cast.golden");
          Golden.graph "sum_of_cast_kernels.golden" (fun () ->
              Golden.sink "sum_of_cast_kernels.golden");
        ];
      prop
        "every number unique_num returns afterwards is greater than each slot \
         and range axis"
        Gen.(
          pair (int_range 1 1000)
            (list ~size:(int_range 1 3) (int_range 1 1000)))
        takes_numbers;
      test "raises Failure on an empty text" (fun () ->
          raises_match (Exn.failure ~sub:"no node") (fun () ->
              Graph.of_string ""));
      cases "raises Failure naming the node and what is wrong with it"
        ~id:(fun (name, _, _, _) -> name)
        faults
        (fun (_, line, fault, text) ->
          raises_match (names_node line fault) (fun () -> Graph.of_string text));
      prop "reading damaged text raises only Failure" case reads_damaged;
    ]

(* cached *)

let clear table =
  List.iter Sys.remove (Disk_cache.entries ~table Helpers.cachedb)

(* What [valid] accepts. *)
type validity = Every | No | Constants

let valid = function
  | Every -> fun _ -> true
  | No -> fun _ -> false
  | Constants -> fun u -> Ops.op u = Const

let pp_validity ppf v =
  Format.pp_print_string ppf
    (match v with
    | Every -> "every graph"
    | No -> "no graph"
    | Constants -> "constants")

let graphs = [ ("3", int 3); ("x", x); ("buffer", buffer); ("n", n) ]

let a_graph =
  Gen.map snd
    (Gen.of_list
       ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
       graphs)

(* The model of a table: each key's graph, and whether it was made anew in place
   of a graph [valid] refused. *)
type entry = { graph : Ops.t; replaced : bool }

let model_cached (m : (string, entry) Hashtbl.t) key v g =
  let kept = Hashtbl.find_opt m key in
  let hit = match kept with Some e -> valid v e.graph | None -> false in
  cover "a hit" hit;
  cover "a miss of an absent key" (Option.is_none kept);
  cover "a miss that valid refused" (Option.is_some kept && not hit);
  classify "a hit after a replacement"
    (match kept with Some e -> hit && e.replaced | None -> false);
  match kept with
  | Some e when hit -> (e.graph, true)
  | _ ->
      Hashtbl.replace m key { graph = g; replaced = Option.is_some kept };
      (g, false)

let tables = ref 0

(* A table no other case holds, emptied of what an earlier run left. *)
let fresh_table () =
  incr tables;
  let table = Printf.sprintf "test_graph_cached_%d" !tables in
  clear table;
  table

let cache = abstract ~release:clear "c"

let model_commands =
  [
    command "create"
      (Gen.unit @-> makes cache)
      (fun () -> Hashtbl.create 2)
      fresh_table;
    command "cached"
      (cache
      ^-> Gen.with_pp Format.pp_print_string
            (Gen.frequency [ (3, Gen.constant "a"); (1, Gen.constant "b") ])
      @-> Gen.with_pp pp_validity
            (Gen.frequency
               [
                 (3, Gen.constant Every);
                 (1, Gen.constant No);
                 (1, Gen.constant Constants);
               ])
      @-> Gen.with_pp Ops.pp a_graph
      @-> returns (pair node bool))
      model_cached
      (fun table key v g ->
        Graph.cached ~table ~key ~valid:(valid v) (fun () -> g));
  ]

(* A graph whose entry on disk is damaged reads back as itself, or is made anew.
   The lengths of an entry damaged in its value's numbers are stated anew, so
   that they do not show it. *)
let damaged_entry_table = "test_graph_damaged"

let made_anew_if_damaged (u, d) =
  classify_damage d;
  let table = damaged_entry_table in
  let cached make = Graph.cached ~table ~key:"k" ~valid:(fun _ -> true) make in
  clear table;
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
  Out_channel.with_open_bin entry (fun oc -> output_string oc e');
  let made = Ops.sink [] in
  match cached (fun () -> made) with
  | g, true -> equal node ~msg:"read back" u g
  | g, false -> equal node ~msg:"made anew" made g

let needs_cachedb () =
  if Sys.getenv_opt "CACHEDB" = None then skip ~reason:"CACHEDB is unset" ()

let mode_bound () =
  if Sys.win32 then skip ~reason:"no file mode forbids access" ();
  if Unix.geteuid () = 0 then skip ~reason:"root reads and writes any file" ()

let accept_all _ = true

(* A graph whose entry the process may not read, as in a cache another user
   shares, is made anew. *)
let unreadable_entry () =
  needs_cachedb ();
  mode_bound ();
  let table = "test_graph_unreadable" in
  clear table;
  ignore (Graph.cached ~table ~key:"k" ~valid:accept_all (fun () -> x));
  List.iter
    (fun e -> Unix.chmod e 0)
    (Disk_cache.entries ~table Helpers.cachedb);
  let made = Ops.sink [] in
  equal (pair node bool) (made, false)
    (Graph.cached ~table ~key:"k" ~valid:accept_all (fun () -> made))

(* A first put makes the table's directory, which then takes no new entry. *)
let unwritable_entry () =
  needs_cachedb ();
  mode_bound ();
  let table = "test_graph_unwritable" in
  clear table;
  ignore (Graph.cached ~table ~key:"kept" ~valid:accept_all (fun () -> x));
  let dir =
    Filename.dirname (List.hd (Disk_cache.entries ~table Helpers.cachedb))
  in
  Unix.chmod dir 0o500;
  Fun.protect
    ~finally:(fun () -> Unix.chmod dir 0o700)
    (fun () ->
      raises_match (Exn.sys_error ?sub:None) (fun () ->
          Graph.cached ~table ~key:"new" ~valid:accept_all (fun () -> x)))

(* "make (), which replaces the entry": a later call reads the graph made. *)
let replaced () =
  needs_cachedb ();
  let table = "test_graph_replaced" in
  clear table;
  let cached valid g = Graph.cached ~table ~key:"k" ~valid (fun () -> g) in
  ignore (cached accept_all x);
  equal (pair node bool) (buffer, false) (cached (fun _ -> false) buffer);
  equal (pair node bool) (buffer, true) (cached accept_all n)

let disabled () =
  needs_cachedb ();
  let table = "test_graph_disabled" in
  clear table;
  ignore (Graph.cached ~table ~key:"k" ~valid:accept_all (fun () -> x));
  let made = Ops.sink [] in
  Setting.context
    [ B (Setting.cachelevel, 0) ]
    (fun () ->
      equal (pair node bool) (made, false)
        (Graph.cached ~table ~key:"k" ~valid:accept_all (fun () -> made)))

let negative_size () =
  needs_cachedb ();
  let table = "test_graph_negative_size" in
  clear table;
  Setting.context
    [ B (Setting.cachesize, -1) ]
    (fun () ->
      raises_match (Exn.invalid_arg ?sub:None) (fun () ->
          Graph.cached ~table ~key:"k" ~valid:accept_all (fun () -> x)))

let cached =
  group "cached"
    [
      stateful ~count:60
        "is the kept graph and true where valid accepts it, else made anew and \
         false"
        model_commands;
      test "a graph made anew replaces the entry" replaced;
      prop "an entry that does not read is made anew" case made_anew_if_damaged;
      test "an entry that cannot be read is made anew" unreadable_entry;
      test "raises Sys_error if the entry cannot be written" unwritable_entry;
      test "is made anew and false while the disk cache is disabled" disabled;
      test "raises Invalid_argument if cachesize is negative" negative_size;
    ]

let () = exit (run "Tolk.Graph" [ to_string; of_string; cached ])
