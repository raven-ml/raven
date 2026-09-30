open Windtrap
open Tolk_next

let node = Testable.make ~pp:Ops.pp ~equal:Ops.equal
let int n = Ops.v ~arg:(Const (`Int (Z.of_int n))) Const
let cast dt u = Ops.v ~src:[ u ] ~arg:(Dtype dt) Cast
let x = cast Float32 (int 1)
let half = cast Float16 (int 1)

let n =
  Ops.v
    ~arg:
      (Param
         (Ops.param_arg ~slot:(-1)
            ~vmin_vmax:(`Int Z.zero, `Int (Z.of_int 9))
            ~name:"n" ~addrspace:(Some Alu) Weak_int))
    Param

let buffer =
  Ops.v
    ~arg:(Param (Ops.param_arg ~slot:0 ~size:16 ~device:(Single "CPU") Float32))
    Buffer

(* The laws *)

let goldens =
  group "a golden reads into the graph that writes it"
    (List.map
       (fun file -> Golden.graph file (fun () -> Golden.sink file))
       [ "matmul.golden"; "sum_of_cast.golden"; "sum_of_cast_kernels.golden" ])

(* One node for each kind of argument and tag. *)
let kinds =
  let k = Ops.kernel_info in
  [
    ("an integer", int 3);
    ( "an integer beyond 64 bits",
      Ops.v ~arg:(Const (`Int (Z.shift_left Z.one 200))) Const );
    ("a boolean", Ops.v ~arg:(Const (`Bool true)) Const);
    ("a negative zero", Ops.v ~arg:(Const (`Float (-0.0))) Const);
    ("a NaN", Ops.v ~arg:(Const (`Float Float.nan)) Const);
    ("Invalid", Ops.v ~arg:(Const `Invalid) Const);
    ("a cast", x);
    ("a variable", n);
    ("a buffer", buffer);
    ( "a scalar parameter",
      Ops.v
        ~arg:
          (Param
             (Ops.param_arg ~slot:1 ~addrspace:None ~volatile:true
                ~bind_on_realize:true
                ~bound:(`Int (Z.of_int 4))
                Int32))
        Param );
    ( "a range",
      Ops.v
        ~src:[ int 16 ]
        ~arg:(Range { axis_id = [ 1; 0 ]; axis_type = Reduce })
        Range );
    ( "a reduction",
      Ops.v ~src:[ x ] ~arg:(Reduce { op = Add; num_axes = 1 }) Reduce );
    ( "an allreduce",
      Ops.v ~src:[ x ]
        ~arg:(Allreduce { op = Max; device = Multi [ "CPU:0"; "CPU:1" ] })
        Allreduce );
    ("a copy to one device", Ops.v ~src:[ x ] ~arg:(Device (Single "CPU")) Copy);
    ("a shard", Ops.v ~src:[ x ] ~arg:(Shard 1) Mselect);
    ("a permutation", Ops.v ~src:[ x ] ~arg:(Axes [ 1; 0 ]) Permute);
    ("an unshard of one axis", Ops.v ~src:[ x ] ~arg:(Axes [ 0 ]) Unshard);
    ("a flip", Ops.v ~src:[ x ] ~arg:(Flips [ true; false ]) Flip);
    ("a launch dimension", Ops.v ~src:[ int 4 ] ~arg:(String "gidx0") Special);
    ( "a flagged load",
      Ops.v ~src:[ Ops.index buffer [ int 0 ] ] ~arg:(String "nontemporal") Load
    );
    ( "a source with every escape",
      Ops.v ~arg:(String "a\"b\\c\nd\te\x00\xc3\xa9") Source );
    ("bytes", Ops.v ~arg:(Bytes "\x7fELF\x00") Binary);
    ( "a queue",
      Ops.v ~arg:(Queue { devices = [ "AMD" ]; queue = "compute" }) Linear );
    ( "custom code",
      Ops.v ~src:[ x ] ~arg:(Code { code = "{0}+1"; dtype = Float32 }) Custom );
    ( "a stage",
      Ops.v ~src:[ buffer ]
        ~arg:
          (Bufferize
             {
               device = Some (Single "CPU");
               addrspace = Local;
               removable = false;
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
               vars = [ n ];
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
          (Call
             { name = Some "f"; precompile = true; aux = None; dtype = Float32 })
        Call );
    ( "a call that submits command queues",
      let kernel : Ops.hcq_kernel =
        {
          devices = [ "AMD" ];
          name = "E_4";
          estimates = { ops = Int 4; lds = Int 32; mem = Int 32 };
          stamps = [ 3; 4 ];
          profile_key = Some "\x00\xff key";
          input_slots = [ 0; 1 ];
          outs = [ 0 ];
          ins = [ 1 ];
        }
      in
      let queues : Ops.hcq_info =
        {
          device = [ "AMD" ];
          kernels = [ kernel ];
          estimates = { ops = Sym n; lds = Int 0; mem = Int 0 };
          nargs = 3;
          table = 2;
          inputs = [ (buffer, 0, "GLOBAL") ];
          slots = [ ("AMD", 1) ];
          host_deps = [ ("CPU", "AMD") ];
          written_bufs = [ buffer ];
          skip_wait = true;
        }
      in
      Ops.v
        ~src:[ Ops.sink [ x ]; buffer ]
        ~arg:
          (Call
             {
               name = None;
               precompile = false;
               aux = Some queues;
               dtype = Void;
             })
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
                 Some ([ ([ 0 ], 2) ], [ ([ 1 ], 2) ], [ ([ 2 ], 2) ]);
             })
        Wmma );
    ("a tuple tag", Ops.v ~src:[ x ] ~tag:(Tuple [ Int 0; Dtype Int32 ]) Add);
    ("a bytes tag", Ops.rtag ~tag:(Bytes "\x01") x);
    ("a string tag", Ops.rtag ~tag:(String "mergeable") x);
    ( "a chain of 100000 nodes",
      List.fold_left
        (fun u _ -> Ops.v ~src:[ u ] Neg)
        x (List.init 100_000 Fun.id) );
  ]

let round_trip =
  cases ~name:fst "a graph reads back as itself" kinds (fun (_, u) ->
      equal node u (Graph.of_string (Graph.to_string u)))

(* The text *)

let text =
  group "text"
    [
      test "writes one line per node, sources first" (fun () ->
          equal Windtrap.text
            "0 Ops.CONST dtypes.weakint [] 1\n\
             1 Ops.CAST dtypes.float [0] dtypes.float\n\
             2 Ops.ADD dtypes.float [1, 1] tag=(0, dtypes.int)\n"
            (Graph.to_string
               (Ops.v ~src:[ x; x ] ~tag:(Tuple [ Int 0; Dtype Int32 ]) Add)));
      test "writes a record's fields that differ from their default" (fun () ->
          equal Windtrap.text
            "0 Ops.BUFFER dtypes.float [] ParamArg(slot=0, dtype=dtypes.float, \
             size=16, device=\"CPU\")\n"
            (Graph.to_string buffer));
      test "writes an argument's nodes before the node that holds them"
        (fun () ->
          let program = List.assoc "a program" kinds in
          equal Windtrap.text
            "0 Ops.PARAM dtypes.weakint [] ParamArg(slot=-1, \
             dtype=dtypes.weakint, vmin_vmax=(0, 9), name=\"n\", \
             addrspace=AddrSpace.ALU)\n\
             1 Ops.PROGRAM dtypes.void [] ProgramInfo(global_size=(%0, 1, 1), \
             vars=(%0,), globals=(0,), outs=(0,), \
             target=Target(device=\"CPU\", renderer=\"CLANG\"))\n"
            (Graph.to_string program));
    ]

(* The errors *)

let rejects substring text =
  raises_match (Exn.failure ~substring) (fun () -> Graph.of_string text)

let errors =
  group "reading"
    [
      test "rejects an empty text" (fun () -> rejects "no node" "");
      test "rejects a written dtype that the node does not derive" (fun () ->
          rejects
            "node 1: the node's data type is dtypes.float, not the written \
             dtypes.int"
            "0 Ops.CONST dtypes.weakint [] 1\n\
             1 Ops.CAST dtypes.int [0] dtypes.float\n");
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
          rejects "a split cannot make an axis of type AxisType.REDUCE"
            "0 Ops.SINK dtypes.void [] \
             KernelInfo(applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, \
             AxisType.REDUCE)),))\n");
      test "rejects an unknown field" (fun () ->
          rejects "KernelInfo has no field size"
            "0 Ops.SINK dtypes.void [] KernelInfo(size=1)\n");
      test "rejects aux data that is not a queue call's" (fun () ->
          rejects "expected a HCQInfo, found an integer"
            "0 Ops.CALL dtypes.void [] CallInfo(aux=1)\n");
      test "rejects text after the argument that is not a tag" (fun () ->
          rejects "expected \" tag=\" at column 32"
            "0 Ops.CONST dtypes.weakint [] 1 2\n");
      test "rejects text after the tag" (fun () ->
          rejects "unexpected text at column 38"
            "0 Ops.CONST dtypes.weakint [] 1 tag=1 2\n");
    ]

let () = exit (run "Graph" [ goldens; round_trip; text; errors ])
