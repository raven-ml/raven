(* Construction to Calls, and elementwise arithmetic: the graphs the
   constructors build. *)

open Windtrap
open Tolk
open Common

let str pp x = Format.asprintf "%a" pp x
let cpu = "CPU"

let param ?device shape slot dt =
  Shape.param ?device ~shape:(ints shape) slot dt

let axis_type = Testable.make ~pp:Ops.Axis_type.pp ~equal:Ops.Axis_type.equal

(* Goldens *)

(* [golden_graph claim file build] is the group [claim] holding the test that
   [build] makes the graph golden [file]. *)
let golden_graph claim file build = group claim [ Golden.graph file build ]

(* [golden value claim file build] is the group [value] › [claim] holding the
   test that [build] makes the graph golden [file]. *)
let golden value claim file build =
  group value [ golden_graph claim file build ]

(* Construction *)

(* [accepted pred rows] is the names of the rows whose node [pred] holds of. *)
let accepted pred rows =
  List.filter_map (fun (what, u) -> if pred u then Some what else None) rows

(* A constant of any kind, an integer of any magnitude, a float of any bits or a
   boolean, of a type that is no literal's three times in four. *)
let gen_const =
  Gen.frequency
    [
      (2, Gen.map (fun n -> `Int n) Dtypes.integer);
      (1, Gen.map (fun x -> `Float x) Gen.any_float);
      (1, Gen.map (fun x -> `Float x) Dtypes.finite_float);
      (1, Gen.map (fun b -> `Bool b) Gen.bool);
    ]

let literal_types = Dtype.[ Weak_int; Weak_float; Bool ]

let gen_typed_const =
  Gen.pair
    (Gen.frequency
       [
         (1, Gen.of_list ~pp:(Testable.pp dtype) literal_types);
         (3, Dtypes.promotable);
       ])
    (Gen.with_pp Dtype.pp_const gen_const)

let literal c = Ops.v ~arg:(Const c) Op.Const

let const_group =
  group "const"
    [
      golden_graph "builds each constant as typed_constants.golden states"
        "typed_constants.golden" Goldens.typed_constants;
      cases "without a type, or of Invalid, is the weak literal of its kind"
        ~id:(fun (what, _, _) -> what)
        [
          ("an integer", None, i 3);
          ("a float", None, f 3.);
          ("a boolean", None, `Bool true);
          ("Invalid", None, `Invalid);
          ("Invalid of a float type", Some Dtype.Float32, `Invalid);
        ]
        (fun (_, dtype, c) -> equal uop (literal c) (Ops.const ?dtype c));
      prop
        "of another type is the cast of the literal of its value converted to \
         that type"
        gen_typed_const (fun (dt, c) ->
          match Dtype.const dt c with
          | exception Invalid_argument _ ->
              (* No value converts: an infinity or a NaN to a weak integer. *)
              rejects (fun () -> Ops.const ~dtype:dt c)
          | converted ->
              let lit = literal converted in
              let is_literal = Dtype.equal (Ops.dtype lit) dt in
              cover "a type of the literal" is_literal;
              cover "another type" (not is_literal);
              let expected =
                if is_literal then lit
                else Ops.v ~src:[ lit ] ~arg:(Dtype dt) Op.Cast
              in
              equal uop expected (Ops.const ~dtype:dt c));
    ]

let gen_number =
  Gen.triple Dtypes.promotable Gen.int (Gen.pair Gen.any_float Gen.bool)

let literal_makers =
  group "int, float, bool and invalid"
    [
      prop "are const of their value" gen_number (fun (dt, n, (x, b)) ->
          let built f =
            match f () with
            | u -> Ok u
            | exception Invalid_argument _ -> Error ()
          in
          let same f g = equal (result uop unit) (built f) (built g) in
          same
            (fun () -> Ops.const ~dtype:dt (`Int (Bigint.of_int n)))
            (fun () -> Ops.int ~dtype:dt n);
          same
            (fun () -> Ops.const ~dtype:dt (`Float x))
            (fun () -> Ops.float ~dtype:dt x);
          same
            (fun () -> Ops.const ~dtype:dt (`Bool b))
            (fun () -> Ops.bool ~dtype:dt b);
          equal uop (Ops.const `Invalid) Ops.invalid);
    ]

let value_group =
  group "value"
    [
      prop "of a constant is its value, read through a cast" gen_typed_const
        (fun (dt, c) ->
          match Dtype.const dt c with
          | exception Invalid_argument _ -> reject ()
          | converted ->
              equal const converted (Ops.value (Ops.const ~dtype:dt c)));
      cases "raises for a node that is neither a constant nor a cast of one"
        ~id:fst
        [
          ("a variable", var "a" 0 1);
          ("a cast of a variable", Ops.cast (var "a" 0 1) Float32);
          ("a sum of constants", Ops.O.(int 1 + int 2));
        ]
        (fun (_, u) -> rejects (fun () -> Ops.value u));
    ]

let is_literal_group =
  group "is_literal"
    [
      test "is true of a constant and a cast of one" (fun () ->
          equal (list string)
            [ "a weak integer"; "a 16 of Int32"; "Invalid"; "a cconst" ]
            (accepted Ops.is_literal
               [
                 ("a weak integer", Ops.int 16);
                 ("a 16 of Int32", Ops.int ~dtype:Int32 16);
                 ("Invalid", Ops.invalid);
                 ("a cconst", Ops.cconst Bool (`Bool true));
                 ("a variable", var "a" 0 1);
                 ("a cast of a variable", Ops.cast (var "a" 0 1) Float32);
                 ("a negated constant", Ops.v ~src:[ Ops.int 1 ] Op.Neg);
               ]));
    ]

let is_invalid_group =
  group "is_invalid"
    [
      test "is true of the constant Invalid alone" (fun () ->
          equal (list string) [ "Invalid" ]
            (accepted Ops.is_invalid
               [
                 ("Invalid", Ops.invalid);
                 ("false", Ops.bool false);
                 ("0", Ops.int 0);
                 ("a validity", Shape.valid (Ops.int 1) (flag "p"));
                 ("a stack of Invalid", Ops.v ~src:[ Ops.invalid ] Op.Stack);
               ]));
    ]

let ccast_group =
  group "ccast"
    [
      golden_graph "builds each constant cast as ccast.golden states"
        "ccast.golden" Goldens.ccast;
    ]

let cconst_group =
  group "cconst"
    [
      prop "is the cast of the literal, even to the literal's own type"
        gen_typed_const (fun (dt, c) ->
          equal uop
            (Ops.v ~src:[ literal c ] ~arg:(Dtype dt) Op.Cast)
            (Ops.cconst dt c));
    ]

let sink_group =
  let a = var "a" 0 1 and b = var "b" 0 1 in
  group "sink"
    [
      test "is the Sink of its nodes, in order" (fun () ->
          equal uop (Ops.v ~src:[ b; a; b ] Op.Sink) (Ops.sink [ b; a; b ]));
      test "with a kernel, is that kernel's root" (fun () ->
          let k = Ops.kernel_info ~name:"k" () in
          equal uop
            (Ops.v ~src:[ a ] ~arg:(Kernel k) Op.Sink)
            (Ops.sink ~kernel:k [ a ]));
    ]

let kernel_nodes_group =
  group "group, index, end_, backedge, after, barrier, range and loop"
    [
      golden_graph "build each node as kernel_nodes.golden states"
        "kernel_nodes.golden" Goldens.kernel_nodes;
    ]

(* An end of a store over a range: an effect. *)
let ended_store () =
  let r = Ops.range ~axis_type:Weak (Int 10) [ 0 ] in
  Ops.end_
    (Ops.store
       (Ops.index (param [ 10 ] 0 Float32) [ r ])
       (Ops.float ~dtype:Float32 1.))
    [ r ]

let group_group =
  group "group"
    [
      test "of one node is that node" (fun () ->
          let a = var "a" 0 1 in
          equal uop a (Ops.group [ a ]));
      test "of several nodes is the empty stack ordered after them" (fun () ->
          let e = ended_store () in
          equal uop (Ops.after (Ops.v Op.Stack) [ e; e ]) (Ops.group [ e; e ]));
      test "of one effect twice simplifies to the empty stack after it once"
        (fun () ->
          let e = ended_store () in
          equal uop
            (Ops.after (Ops.v Op.Stack) [ e ])
            (Shape.simplify (Ops.group [ e; e ])));
    ]

let is_group_group =
  group "is_group"
    [
      test "is true of an After of the empty Stack" (fun () ->
          let e = ended_store () and p = param [ 4 ] 1 Float32 in
          equal (list string)
            [ "a group of two effects"; "an After of the empty Stack alone" ]
            (accepted Ops.is_group
               [
                 ("a group of two effects", Ops.group [ e; e ]);
                 ("a group of one effect", Ops.group [ e ]);
                 ("the empty Stack", Ops.v Op.Stack);
                 ( "an After of the empty Stack alone",
                   Ops.v ~src:[ Ops.v Op.Stack ] Op.After );
                 ( "an After of a Stack of one node",
                   Ops.after (Ops.v ~src:[ Ops.int 1 ] Op.Stack) [ e ] );
                 ("an After of storage", Ops.after p [ e ]);
               ]));
    ]

let broadcast_group =
  let a = var "a" 0 1 in
  group "broadcast"
    [
      cases "is the Stack of n copies, or the node if n is 1"
        ~id:(fun (n, _) -> string_of_int n)
        [
          (0, Ops.v Op.Stack);
          (1, a);
          (2, Ops.v ~src:[ a; a ] Op.Stack);
          (5, Ops.v ~src:[ a; a; a; a; a ] Op.Stack);
        ]
        (fun (n, expected) -> equal uop expected (Ops.broadcast a n));
    ]

(* A stack of [n] distinct variables, from 1 to 5, and a literal position in it
   or just outside: [-n - 1] to [n]. *)
let gen_stack_index =
  Gen.(
    let* n = int_range 1 5 in
    let+ k = int_range (-n - 1) n in
    (n, k))

let index_group =
  group "index"
    [
      prop
        "of a stack by a weak integer literal is that source, a negative one \
         counting from the end"
        gen_stack_index (fun (n, k) ->
          let srcs = List.init n (fun j -> var (Printf.sprintf "v%d" j) 0 1) in
          let s = Ops.v ~src:srcs Op.Stack in
          let inside = -n <= k && k < n in
          cover "a position from the start" (0 <= k && k < n);
          cover "a position from the end" (-n <= k && k < 0);
          cover "a position outside" (not inside);
          if inside then
            equal uop
              (List.nth srcs ((k + n) mod n))
              (Ops.index s [ Ops.int k ])
          else rejects (fun () -> Ops.index s [ Ops.int k ]));
      cases "of anything else is the Index of the node by the indices"
        ~id:(fun (what, _, _) -> what)
        (let s = Ops.v ~src:[ var "a" 0 1; var "b" 0 1 ] Op.Stack in
         let p = param [ 4 ] 0 Float32 in
         [
           ("a stack by an Int32 constant", s, [ Ops.int ~dtype:Int32 0 ]);
           ("a stack by a variable", s, [ var "i" 0 1 ]);
           ("a stack by two literals", s, [ Ops.int 0; Ops.int 1 ]);
           ("storage by a literal", p, [ Ops.int 3 ]);
         ])
        (fun (_, u, idxs) ->
          equal uop (Ops.v ~src:(u :: idxs) Op.Index) (Ops.index u idxs));
    ]

let end_group =
  group "end_"
    [
      test "of no range is the node" (fun () ->
          let a = var "a" 0 1 in
          equal uop a (Ops.end_ a []));
    ]

let after_group =
  group "after"
    [
      test "of no dependency is the node" (fun () ->
          let a = var "a" 0 1 in
          equal uop a (Ops.after a []));
    ]

let after_disjoint_group =
  group "after_disjoint"
    [
      test "is after of the stores, marked disjoint" (fun () ->
          let p = param [ 10 ] 0 Float32 and e = ended_store () in
          let d = Ops.after_disjoint p [ e; e ] in
          equal (pair op uops)
            (Ops.op (Ops.after p [ e; e ]), Ops.src (Ops.after p [ e; e ]))
            (Ops.op d, Ops.src d);
          equal arg Disjoint (Ops.arg d));
    ]

(* Storage, effects over it and afters of each, for the predicates of afters. *)
let after_rows () =
  let p = param [ 10 ] 0 Float32 and q = param [ 10 ] 1 Float32 in
  let e = ended_store () in
  let state = Ops.after q [ e ] in
  [
    ("a disjoint write", Ops.after_disjoint p [ e; e ]);
    ("a write", Ops.after p [ e; e ]);
    ("a join", Ops.after p [ state ]);
    ("a join of two states", Ops.after p [ state; Ops.after q [ e; e ] ]);
    ("an after of a state and an effect", Ops.after p [ state; e ]);
    ("a group of states", Ops.group [ state; Ops.after p [ e ] ]);
    ("storage", p);
  ]

let is_disjoint_group =
  group "is_disjoint"
    [
      test "is true of the After of an after_disjoint alone" (fun () ->
          equal (list string) [ "a disjoint write" ]
            (accepted Ops.is_disjoint (after_rows ())));
    ]

let is_join_group =
  group "is_join"
    [
      test "is true of an After whose dependencies are all Afters" (fun () ->
          equal (list string)
            [ "a join"; "a join of two states"; "a group of states" ]
            (accepted Ops.is_join (after_rows ())));
    ]

let without_after_group =
  group "without_after"
    [
      test "is the node without its Afters" (fun () ->
          let a = var "a" 0 1 and e = ended_store () in
          equal uops [ a; a ]
            [
              Ops.without_after (Ops.after (Ops.after a [ e ]) [ e ]);
              Ops.without_after a;
            ]);
    ]

let ins_group =
  group "ins"
    [
      golden_graph "builds each instruction as instructions.golden states"
        "instructions.golden" Goldens.instructions;
    ]

let range_group =
  group "range"
    [
      test "is of a weak integer type unless given, its end its source"
        (fun () ->
          let r = Ops.range ~axis_type:Loop (Int 4) [ 0 ] in
          equal
            (triple dtype axis_type uops)
            (Weak_int, Loop, [ Ops.int 4 ])
            (Ops.dtype r, Ops.axis_type r, Ops.src r));
    ]

let loop_group =
  group "loop"
    [
      test "is an unbounded void range of identity [axis]" (fun () ->
          let l = Ops.loop 2 in
          equal
            (triple dtype (list int) op)
            (Void, [ 2 ], Op.Noop)
            (Ops.dtype l, Ops.axis_id l, Ops.op (Ops.nth l 0)));
    ]

let wmma_group =
  group "wmma"
    [
      golden_graph "builds each product as wmmas.golden states" "wmmas.golden"
        Goldens.wmmas;
    ]

let validity_group =
  group "get_idx and get_valid"
    [
      golden_graph "split each validity as validity.golden states"
        "validity.golden" Goldens.validity;
    ]

(* Placement *)

let device_group =
  group "device"
    [
      cases
        "is its argument's, its first source's for an After or an unplaced \
         stage, and its sources' first otherwise"
        ~id:(fun (what, _, _) -> what)
        (let p = param ~device:cpu [ 4 ] 0 Float32 in
         let unplaced = param [ 4 ] 1 Float32 in
         let opts : Ops.bufferize_opts =
           { device = Some "CUDA"; addrspace = Global; keep = Removable }
         in
         [
           ("placed storage", p, Some cpu);
           ("unplaced storage", unplaced, None);
           ("a copy", Ops.copy_to_device p "CUDA", Some "CUDA");
           ("a stage placed on CUDA", Ops.bufferize ~opts p [], Some "CUDA");
           ("an unplaced stage", Ops.bufferize p [], Some cpu);
           ( "an After of unplaced storage, after placed storage",
             Ops.after unplaced [ p ],
             None );
           ("a sum of placed storage", Ops.O.(p + float 1.), Some cpu);
           ( "a sum of unplaced, then placed storage",
             Ops.O.(param [ 4 ] 1 Float32 + p),
             Some cpu );
           ( "a sum of unplaced storage",
             Ops.O.(param [ 4 ] 1 Float32 + float 1.),
             None );
           ("a variable", var "a" 0 1, None);
         ])
        (fun (_, u, expected) -> equal (option device) expected (Ops.device u));
      test "of a deep graph needs no deep recursion" (fun () ->
          let rec deepen n u =
            if n = 0 then u else deepen (n - 1) Ops.O.(u + u)
          in
          equal (option device) (Some cpu)
            (Ops.device (deepen 10_000 (Ops.new_buffer ~slot:0 cpu 1 Int8))));
    ]

let on_disk_group =
  group "on_disk"
    [
      test "is true of a node on a disk device" (fun () ->
          equal (list string)
            [ "storage on DISK:/tmp/f"; "storage on disk:0"; "storage on DiSk" ]
            (accepted Ops.on_disk
               [
                 ( "storage on DISK:/tmp/f",
                   param ~device:"DISK:/tmp/f" [ 4 ] 0 Uint8 );
                 ("storage on disk:0", param ~device:"disk:0" [ 4 ] 0 Uint8);
                 ("storage on DiSk", param ~device:"DiSk" [ 4 ] 0 Uint8);
                 ("storage on DISKS", param ~device:"DISKS" [ 4 ] 0 Uint8);
                 ("storage on AMD:DISK", param ~device:"AMD:DISK" [ 4 ] 0 Uint8);
                 ("storage on CPU", param ~device:cpu [ 4 ] 0 Uint8);
                 ("unplaced storage", param [ 4 ] 0 Uint8);
               ]));
    ]

let copy_to_device_group =
  let p = param ~device:cpu [ 4 ] 0 Float32 in
  group "copy_to_device"
    [
      test "is the Copy of the node to the device" (fun () ->
          equal uop
            (Ops.v ~src:[ p ] ~arg:(Device "CUDA") Op.Copy)
            (Ops.copy_to_device p "CUDA"));
      cases "raises for a disk or a weak type"
        ~id:(fun (what, _, _) -> what)
        [
          ("to DISK:/tmp/f", p, "DISK:/tmp/f");
          ("to disk", p, "disk");
          ("to disk:0", p, "disk:0");
          ("of a weak integer", Ops.int 1, cpu);
          ("of a weak float", Ops.float 1., cpu);
        ]
        (fun (_, u, d) -> rejects (fun () -> Ops.copy_to_device u d));
    ]

let size_src_group =
  group "size_src"
    [
      test "is the weak integer of the size, or the empty Stack for a scalar"
        (fun () ->
          equal uops
            [ Ops.int 12; Ops.v Op.Stack ]
            [ Ops.size_src (Some 12); Ops.size_src None ]);
    ]

(* Movement *)

let base_group =
  let p = Ops.base (param [ 2; 3; 4 ] 0 Float32) in
  group "base"
    [
      cases "is the node without its movements"
        ~id:(fun (what, _, _) -> what)
        (let s = Ops.bufferize (param ~device:cpu [ 2; 4 ] 1 Float32) [] in
         [
           ("storage", p, p);
           ( "a permute of a reshape",
             Shape.permute (Shape.reshape p (ints [ 6; 4 ])) [ 1; 0 ],
             p );
           ("a stage", s, s);
         ])
        (fun (_, u, expected) -> equal uop expected (Ops.base u));
    ]

let storage_base_group =
  group "storage_base"
    [
      test "is the base without bitcasts and Afters" (fun () ->
          let p = param ~device:cpu [ 4 ] 2 Uint32 in
          let viewed =
            Shape.reshape
              (Ops.bitcast (Ops.after p [ Ops.sink [] ]) Int32)
              (ints [ 2; 2 ])
          in
          equal uop p (Ops.storage_base viewed));
    ]

(* Storage *)

let addrspace_group =
  let local = Shape.placeholder ~slot:0 ~addrspace:Local [ 8 ] Float32 in
  let at n = Ops.index local [ Ops.int n ] in
  group "addrspace"
    [
      cases "is where storage is placed, and where a value's memory lives"
        ~id:(fun (what, _, _) -> what)
        [
          ("workgroup storage", local, Some Dtype.Local);
          ("unplaced storage", param [ 2; 2 ] 1 Float32, Some Global);
          ("binary data", Ops.v ~arg:(Bytes "a") Op.Binary, Some Global);
          ("an index of workgroup storage", at 0, Some Local);
          ("a load", Ops.load (at 0) [], Some Alu);
          ("a variable", var "a" 0 1, Some Alu);
          ("a hardware index", Ops.special (Int 4) "gidx0", Some Alu);
          ( "a store, of type Void",
            Ops.store (at 0) (Ops.float ~dtype:Float32 1.),
            Some Alu );
          ("a sink, of type Void", Ops.sink [], Some Alu);
          ( "a stack of indices of workgroup storage",
            Ops.v ~src:[ at 0; at 1 ] Op.Stack,
            Some Local );
          ( "a stack of workgroup and global indices",
            Ops.v ~src:[ at 0; param [ 1 ] 1 Float32 ] Op.Stack,
            None );
          ( "a sum of workgroup storage and a constant",
            Ops.O.(local + Ops.int ~dtype:Float32 1),
            None );
        ]
        (fun (_, u, expected) ->
          equal (option addr_space) expected (Ops.addrspace u));
    ]

let whole_group =
  group "whole"
    [
      cases "is a view's storage, and any other node itself"
        ~id:(fun (what, _, _) -> what)
        (let p = param ~device:cpu [ 8 ] 0 Float32 in
         let view = Ops.replace ~src:[ Ops.size_src (Some 4); p ] p in
         [
           ("a view", view, p);
           ("storage", p, p);
           ("a variable", var "a" 0 1, var "a" 0 1);
           ( "an index of a view",
             Ops.index view [ Ops.int 0 ],
             Ops.index view [ Ops.int 0 ] );
         ])
        (fun (_, u, expected) -> equal uop expected (Ops.whole u));
    ]

let buf_uop_group =
  group "buf_uop"
    [
      cases "is the storage a node accesses"
        ~id:(fun (what, _, _) -> what)
        (let p = param [ 4 ] 0 Float32 in
         let s =
           Ops.contiguous Ops.O.(param ~device:cpu [ 4 ] 1 Float32 + float 1.)
         in
         [
           ("storage", p, p);
           ("an index of storage", Ops.index p [ Ops.int 0 ], p);
           ("a stage", s, s);
         ])
        (fun (_, u, expected) -> equal uop expected (Ops.buf_uop u));
    ]

let is_virtual_group =
  group "is_virtual"
    [
      test "is true of a node without a device or of a weak type" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          equal (list string)
            [ "unplaced storage"; "a weak cast of placed storage" ]
            (accepted Ops.is_virtual
               [
                 ("unplaced storage", param [ 4 ] 0 Float32);
                 ("placed storage", p);
                 ("a weak cast of placed storage", Ops.cast p Weak_float);
               ]));
    ]

let has_buffer_identity_group =
  group "has_buffer_identity"
    [
      test "is true of storage through reshapes and, if asked, an After"
        (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          let rows =
            [
              ("storage", p);
              ("a reshape", Shape.reshape p (ints [ 2; 2 ]));
              ( "a permute of a reshape",
                Shape.permute (Shape.reshape p (ints [ 2; 2 ])) [ 1; 0 ] );
              ("an After", Ops.after p [ Ops.sink [] ]);
              ("a stage", Ops.bufferize p []);
            ]
          in
          equal
            (pair (list string) (list string))
            ([ "storage"; "a reshape" ], [ "storage"; "a reshape"; "an After" ])
            ( accepted Ops.has_buffer_identity rows,
              accepted (Ops.has_buffer_identity ~after_ok:true) rows ));
    ]

let needs_storage_group =
  group "needs_storage"
    [
      test "is true of a placed value that is no storage" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          equal (list string)
            [ "a sum of placed storage"; "an allocation" ]
            (accepted Ops.needs_storage
               [
                 ("storage", p);
                 ("a sum of placed storage", Ops.O.(p + float 1.));
                 ( "an allocation",
                   Shape.alloc ~slot:3 ~device:cpu (ints [ 4 ]) Float32 );
                 ( "a sum of unplaced storage",
                   Ops.O.(param [ 4 ] 1 Float32 + float 1.) );
               ]));
    ]

(* The numbers [unique_num] returned in a program, on either domain. *)
let rec record drawn n =
  let l = Atomic.get drawn in
  if not (Atomic.compare_and_set drawn l (n :: l)) then record drawn n

let distinct () = function
  | Ok ns ->
      equal (list int)
        (List.sort_uniq Int.compare ns)
        (List.sort Int.compare ns)
  | Error e -> raise e

let unique_num_group =
  let drawn = abstract "d" in
  group "unique_num"
    [
      stateful ~domains:2 "never returns a number twice, from two domains"
        [
          command "drawn"
            (Gen.unit @-> makes drawn)
            ignore
            (fun () -> Atomic.make []);
          command "unique_num"
            (drawn ^-> returns unit)
            ignore
            (fun d -> record d (Ops.unique_num ()));
          command "returned" (drawn ^-> judges (list int)) distinct Atomic.get;
        ];
    ]

let take_unique_group =
  group "take_unique"
    [
      test "makes every later number greater" (fun () ->
          let n = Ops.unique_num () + 1000 in
          Ops.take_unique n;
          greater int ~than:n (Ops.unique_num ()));
      test "raises for max_int" (fun () ->
          rejects (fun () -> Ops.take_unique max_int));
    ]

let slot u = (Ops.param_arg_of u).slot

let new_buffer_group =
  group "new_buffer"
    [
      test
        "is a Buffer of the size and type on the device, in the slot, at the \
         phase" (fun () ->
          let s = Ops.unique_num () in
          let buffer ?phase () =
            Ops.v
              ~src:[ Ops.size_src (Some 8) ]
              ~arg:(Param (Ops.param_arg ~slot:s ~device:cpu ?phase Float32))
              Op.Buffer
          in
          equal uops
            [ buffer ~phase:4 (); buffer ~phase:0 () ]
            [
              Ops.new_buffer ~slot:s ~phase:4 cpu 8 Float32;
              Ops.new_buffer ~slot:s cpu 8 Float32;
            ]);
      test "without a slot, is in a slot of its own" (fun () ->
          not_equal int
            (slot (Ops.new_buffer cpu 4 Float32))
            (slot (Ops.new_buffer cpu 4 Float32)));
      test "given a slot, is no buffer made without one" (fun () ->
          (* The number the next fresh buffer would take, given by hand
             first. *)
          let s = Ops.unique_num () + 1 in
          let given = Ops.new_buffer ~slot:s cpu 8 Float32 in
          let made = Ops.new_buffer cpu 8 Float32 in
          not_equal int (slot given) (slot made));
      test "of one slot, size, type, device and phase is one node" (fun () ->
          let s = Ops.unique_num () in
          equal uop
            (Ops.new_buffer ~slot:s ~phase:4 cpu 8 Float32)
            (Ops.new_buffer ~slot:s ~phase:4 cpu 8 Float32));
      cases "raises for a weak type or the slot max_int"
        ~id:(fun (what, _, _) -> what)
        [
          ("a weak integer", None, Dtype.Weak_int);
          ("a weak float", None, Weak_float);
          ("the slot max_int", Some max_int, Float32);
        ]
        (fun (_, slot, dt) -> rejects (fun () -> Ops.new_buffer ?slot cpu 8 dt));
    ]

let set_group =
  group "set"
    [
      golden_graph "stores through each pointer as sets.golden states"
        "sets.golden" Goldens.sets;
    ]

(* Variables *)

let variable_group =
  group "variable"
    [
      cases "is the scalar variable of the name, type and bounds"
        ~id:(fun (what, _, _) -> what)
        [
          ( "of no type, a weak integer",
            Ops.variable "n" (i 1) (i 8),
            (Dtype.Weak_int, int_bounds 1 8) );
          ( "an Int64",
            Ops.variable ~dtype:Int64 "n" (i (-3)) (i 3),
            (Int64, int_bounds (-3) 3) );
          ( "a Float16",
            Ops.variable ~dtype:Float16 "n" (f (-0.5)) (f 2.),
            (Float16, (f (-0.5), f 2.)) );
        ]
        (fun (_, u, (dt, b)) ->
          equal
            (quad string dtype shape (pair value value))
            ("n", dt, [], b)
            (Ops.expr u, Ops.dtype u, Shape.shape u, bounds u));
    ]

let expr_group =
  group "expr"
    [
      test "is the name of a named parameter" (fun () ->
          equal string "w"
            (Ops.expr (Shape.param ~name:"w" ~shape:(ints [ 4 ]) 0 Float32)));
      test "raises for a node without a name" (fun () ->
          rejects (fun () -> Ops.expr (param [ 4 ] 0 Float32)));
    ]

(* Divisibility *)

(* A sum or product of constants from [-6] to [6] and a weak integer [x] from
   [-4] to [6], at most three operations deep. *)
let x_var = weak_var "x" (-4) 6

let gen_multiple =
  let open Gen in
  let leaf =
    frequency
      [ (1, map (fun () -> x_var) unit); (2, map Ops.int (int_range (-6) 6)) ]
  in
  let rec expr depth =
    if depth = 0 then leaf
    else
      let sub = expr (depth - 1) in
      frequency
        [
          (1, leaf);
          ( 2,
            let+ a = sub and+ b = sub in
            Ops.O.(a + b) );
          ( 2,
            let+ a = sub and+ b = sub in
            Ops.O.(a * b) );
        ]
  in
  with_pp Ops.pp (expr 3)

let divides k (n : Dtype.const) =
  match n with
  | `Int n ->
      if Bigint.equal k Bigint.zero then Bigint.equal n Bigint.zero
      else Bigint.equal (Bigint.rem n k) Bigint.zero
  | _ -> false

let const_factor_group =
  let x = weak_var "x" 10 20 in
  group "const_factor"
    [
      (* The values tinygrad's TestConstFactor and test_const_factor state. *)
      cases "is a known divisor"
        ~id:(fun (what, _, _) -> what)
        (let g = Ops.special (Int 8) "gidx0" in
         [
           ("42", Ops.int 42, 42);
           ("30 + 12", Ops.O.(int 30 + int 12), 6);
           ("5 * 7", Ops.O.(int 5 * int 7), 5);
           ("x * 3", Ops.O.(x * int 3), 3);
           ("x // 4", Ops.O.(x // int 4), 1);
           ("gidx0", g, 1);
           ("gidx0 * 3", Ops.O.(g * int 3), 3);
           ("gidx0 * 3 + 6", Ops.O.((g * int 3) + int 6), 3);
           ("gidx0 * 3 + 1", Ops.O.((g * int 3) + int 1), 1);
           ("storage", Shape.param ~shape:(ints [ 4 ]) 0 Int32, 1);
         ])
        (fun (_, u, k) -> equal z (Bigint.of_int k) (Ops.const_factor u));
      (* Weak integers never wrap. *)
      prop "divides every value of a weak integer" gen_multiple (fun u ->
          let k = Ops.const_factor u in
          for v = -4 to 6 do
            let n = Interpreter.eval ~vars:[ ("x", i v) ] u in
            satisfies
              ~claim:(Format.asprintf "a multiple of %a" Bigint.pp_print k)
              const (divides k) n
          done);
    ]

let pop_const_group =
  let x = weak_var "x" 10 20 in
  group "pop_const"
    [
      cases "splits off a constant operand, or the operation's identity"
        ~id:(fun (what, _, _, _) -> what)
        (let e = Ops.O.(x + int 3) in
         [
           ("x + 3", None, e, (x, i 3));
           ("x * 3 by Mul", Some Op.Mul, Ops.O.(x * int 3), (x, i 3));
           ("x + 3 by Mul", Some Op.Mul, e, (e, i 1));
           ("x", None, x, (x, i 0));
         ])
        (fun (_, op, u, expected) ->
          equal (pair uop const) expected (Ops.pop_const ?op u));
    ]

(* Calls *)

let body = Ops.sink []
let call_arg = param ~device:cpu [ 4 ] 0 Float32

let body_group =
  group "body"
    [
      test "is the body a call calls" (fun () ->
          equal uop body (Ops.body (Ops.call body [ call_arg ])));
      test "raises for a node that is not a call" (fun () ->
          rejects (fun () -> Ops.body body));
    ]

let src_without_body_group =
  group "src_without_body"
    [
      cases "is a call's arguments, and any other node's sources"
        ~id:(fun (what, _) -> what)
        [
          ("a call", Ops.call body [ call_arg ]);
          ("a sink", Ops.sink [ call_arg ]);
        ]
        (fun (_, u) -> equal uops [ call_arg ] (Ops.src_without_body u));
    ]

let opaque_call_bodies_group =
  group "opaque_call_bodies"
    [
      test "is the operations a call body can have" (fun () ->
          equal (list op)
            Op.[ Program; Linear; Sink; Store; Custom_function ]
            (Op.Set.to_list Ops.opaque_call_bodies));
    ]

let is_inline_call_group =
  group "is_inline_call"
    [
      test "is true of a call of a plain sink not compiled on its own"
        (fun () ->
          equal (list string) [ "a call of a sink" ]
            (accepted Ops.is_inline_call
               [
                 ("a call of a sink", Ops.call body []);
                 ("a precompiled call", Ops.call ~precompile:true body []);
                 ( "a call of a kernel",
                   Ops.call (Ops.sink ~kernel:(Ops.kernel_info ()) []) [] );
                 ("a sink", body);
               ]));
    ]

let call_with_output () =
  let out = Shape.call_with_output Ops.O.(call_arg + float 1.) [ call_arg ] in
  (out, Ops.nth out 1)

let unbound_outputs_groups =
  [
    group "has_unbound_outputs"
      [
        test "is true of a call with an output no buffer is bound to" (fun () ->
            let _, c = call_with_output () in
            equal (list string)
              [ "a call with an output" ]
              (accepted Ops.has_unbound_outputs
                 [
                   ("a call with an output", c);
                   ("a call of storage", Ops.call body [ call_arg ]);
                   ("storage", call_arg);
                 ]));
      ];
    group "unbound_outputs"
      [
        test "is those outputs, each ordered after the call" (fun () ->
            let out, c = call_with_output () in
            equal uops [ out ] (Ops.unbound_outputs c));
      ];
  ]

let custom_function_group =
  group "custom_function"
    [
      test "is the external function of the name, type and arguments" (fun () ->
          let fn = Ops.custom_function "f" [ call_arg ] in
          equal (triple op arg uops)
            ( Op.Custom_function,
              Custom_function { name = "f"; dtype = Void; fmt = false },
              [ call_arg ] )
            (Ops.op fn, Ops.arg fn, Ops.src fn));
      cases "has no shape when Void, and its shape otherwise"
        ~id:(fun (what, _, _) -> what)
        [
          ("of Void", Ops.custom_function "f" [], None);
          ("of Int32", Ops.custom_function ~dtype:Int32 "f" [], Some []);
          ( "of Int32 and shape [2; 3]",
            Ops.custom_function ~dtype:Int32 ~shape:(ints [ 2; 3 ]) "f" [],
            Some (ints [ 2; 3 ]) );
        ]
        (fun (_, u, expected) ->
          equal (option shape) expected (Shape.shape_opt u));
    ]

let call_group =
  group "call"
    [
      cases "has its body's type and shape"
        ~id:(fun (what, _) -> what)
        [
          ("a function of an int32", Ops.custom_function ~dtype:Int32 "f" []);
          ( "a function of an int32 of shape [2; 3]",
            Ops.custom_function ~dtype:Int32 ~shape:(ints [ 2; 3 ]) "f" [] );
          ("a function of nothing", Ops.custom_function "f" []);
          ("a sink", body);
        ]
        (fun (_, b) ->
          let c = Ops.call b [ call_arg ] in
          equal
            (pair dtype (option shape))
            (Ops.dtype b, Shape.shape_opt b)
            (Ops.dtype c, Shape.shape_opt c));
      cases "raises for a body it cannot call or a range leaking out of it"
        ~id:fst
        (let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
         [
           ("a constant", Ops.int 1);
           ("a sum", Ops.O.(call_arg + float 1.));
           ( "a sink of a store over an open range",
             Ops.sink
               [
                 Ops.store (Ops.index call_arg [ r ])
                   (Ops.float ~dtype:Float32 1.);
               ] );
         ])
        (fun (_, b) -> rejects (fun () -> Ops.call b [ call_arg ]));
    ]

(* Elementwise *)

let dtypes =
  Dtype.
    [
      Bool;
      Int8;
      Uint8;
      Int32;
      Uint32;
      Int64;
      Float16;
      Bfloat16;
      Float32;
      Float64;
    ]

let gen_dtype = Gen.of_list ~pp:(Testable.pp dtype) dtypes

let typed dt name =
  if Dtype.equal dt Bool then flag name
  else Ops.variable ~dtype:dt name (i 0) (i 1)

(* A binary operation of the interface, by name. *)
let binary =
  Ops.
    [
      ("add", add);
      ("sub", sub);
      ("mul", mul);
      ("maximum", maximum);
      ("lt", lt);
      ("ne", ne);
      ("eq", eq);
      ("where", fun x y -> where (flag "c") x y);
    ]

let gen_binary =
  Gen.of_list ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n) binary

let elementwise =
  let a = var "a" 0 10 and b = var "b" 0 10 in
  group "elementwise"
    [
      prop "a binary operation promotes its operands to their least upper type"
        (Gen.triple gen_binary gen_dtype gen_dtype) (fun ((name, fn), d0, d1) ->
          assume (not (Dtype.equal d0 Bool && Dtype.equal d1 Bool));
          let u = fn (typed d0 "x") (typed d1 "y") in
          let promoted = Dtype.least_upper [ d0; d1 ] in
          let expected =
            if name = "lt" || name = "ne" || name = "eq" then Dtype.Bool
            else promoted
          in
          equal dtype expected (Ops.dtype u));
      prop "a weak constant takes the other operand's kind and stays weak"
        gen_dtype (fun dt ->
          assume (not (Dtype.equal dt Bool));
          let u = Ops.O.(typed dt "x" + int 1) in
          equal dtype dt (Ops.dtype u);
          equal dtype (Dtype.weak dt) (Ops.dtype (Ops.nth u 1)));
      test "the operators are the named operations" (fun () ->
          List.iter
            (fun (name, x, y) -> equal ~msg:name uop x y)
            Ops.
              [
                ("+", O.(a + b), add a b);
                ("-", O.(a - b), sub a b);
                ("*", O.(a * b), mul a b);
                ("/", O.(a / b), div a b);
                ("//", O.(a // b), div ~rounding:`Floor a b);
                ("%", O.(a % b), mod_ a b);
                ("~-", O.(~-a), neg a);
                ("<", O.(a < b), lt a b);
                (">", O.(a > b), gt a b);
                ("<=", O.(a <= b), le a b);
                (">=", O.(a >= b), ge a b);
                ("<>", O.(a <> b), ne a b);
                ("land", O.(a land b), bitwise_and a b);
                ("lor", O.(a lor b), bitwise_or a b);
                ("lxor", O.(a lxor b), bitwise_xor a b);
                ("lnot", O.(lnot a), bitwise_not a);
                ("lsl", O.(a lsl b), shl a b);
                ("lsr", O.(a lsr b), shr a b);
                ("int", O.int 3, int 3);
                ("float", O.float 1.5, float 1.5);
                ("bool", O.bool true, bool true);
              ]);
      test "alu applies an operation as given" (fun () ->
          equal uop (Ops.v ~src:[ a; b ] Op.Add) (Ops.alu a Op.Add [ b ]);
          equal uop
            (Ops.v ~src:[ a; Ops.float 1. ] Op.Mul)
            (Ops.alu a Op.Mul [ Ops.float 1. ]));
      test "a cast or bitcast to the node's own type is the node" (fun () ->
          is_true (Ops.cast a Int32 == a);
          is_true (Ops.bitcast a Int32 == a));
      test "a bitcast rejects a weak type on either side" (fun () ->
          rejects (fun () -> Ops.bitcast (Ops.int 1) Int32);
          rejects (fun () -> Ops.bitcast a Weak_int));
      test
        "pow rejects an integer base raised to a constant that is not a natural"
        (fun () ->
          rejects (fun () -> Ops.pow a (Ops.int (-1)));
          ignore (Ops.pow a (Ops.int 0));
          ignore (Ops.pow a (Ops.float 0.5));
          ignore (Ops.pow (fvar "x") (Ops.float 0.5)));
      test "usum and uprod fold booleans with or and and" (fun () ->
          let p = flag "p" and q = flag "q" in
          equal uop (Ops.bitwise_or p q) (Ops.usum p [ q ]);
          equal uop (Ops.bitwise_and p q) (Ops.uprod p [ q ]);
          equal uop Ops.O.(a + b + b) (Ops.usum a [ b; b ]);
          is_true (Ops.uprod a [] == a));
      test
        "commit_dtype commits a weak integer to the narrowest type of its \
         bounds" (fun () ->
          equal dtype Int32 (Ops.commit_dtype (weak_var "n" 0 10));
          equal dtype Int64
            (Ops.commit_dtype
               (Ops.variable ~dtype:Weak_int "n" (i 0)
                  (`Int (Bigint.shift_left Bigint.one 40))));
          equal dtype Int64
            (Ops.commit_dtype ~default_int:Int64 (weak_var "n" 0 10));
          equal dtype Float32 (Ops.commit_dtype (Ops.float 1.));
          equal dtype Float16 (Ops.commit_dtype (fvar ~dtype:Float16 "x")));
      test "element_size is the type's size, and rejects a weak type" (fun () ->
          equal int 2 (Ops.element_size (fvar ~dtype:Float16 "x"));
          equal int 8 (Ops.element_size (var ~dtype:Int64 "n" 0 1));
          rejects (fun () -> Ops.element_size (Ops.int 1)));
      test "contiguous stages a placed value, and is the node otherwise"
        (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          let v = Ops.O.(p + float 1.) in
          equal uop (Ops.v ~src:[ v ] Op.Stage) (Ops.contiguous v);
          is_true (Ops.contiguous p == p);
          is_true (Ops.contiguous (Ops.contiguous v) == Ops.contiguous v);
          is_true (Ops.contiguous (Ops.int 1) == Ops.int 1);
          let unplaced = Ops.O.(param [ 4 ] 1 Float32 + float 1.) in
          is_true (Ops.contiguous unplaced == unplaced));
    ]

let elementwise_goldens =
  [
    golden "sub" "builds each difference as subtraction.golden states"
      "subtraction.golden" Goldens.subtraction;
    golden "neg" "builds each negation as negation.golden states"
      "negation.golden" Goldens.negation;
    golden "add"
      "promotes its operands, a weak constant staying weak, as \
       weak_promotion.golden states"
      "weak_promotion.golden" Goldens.weak_promotion;
    group "div, mod_ and fmod"
      [
        group "build each quotient and remainder as division.golden states"
          [ Golden.graph "division.golden" Goldens.division ];
        group
          "build each quotient and remainder by a constant as \
           constant_division.golden states"
          [ Golden.graph "constant_division.golden" Goldens.constant_division ];
      ];
    golden "lt, gt, le, ge, eq and ne"
      "build each comparison as comparisons.golden states" "comparisons.golden"
      Goldens.comparisons;
    golden "the bitwise operations" "build each node as bitwise.golden states"
      "bitwise.golden" Goldens.bitwise;
    golden "maximum and minimum" "build each node as extrema.golden states"
      "extrema.golden" Goldens.extrema;
    golden "where" "builds each selection as selection.golden states"
      "selection.golden" Goldens.selection;
    golden "the unary operations" "build each node as unary.golden states"
      "unary.golden" Goldens.unary;
    golden "pow" "builds each power as powers.golden states" "powers.golden"
      Goldens.powers;
    golden "usum and uprod"
      "build each sum and product as sums_and_products.golden states"
      "sums_and_products.golden" Goldens.sums_and_products;
    golden "cast" "builds each cast as casts.golden states" "casts.golden"
      Goldens.casts;
    golden "bitcast" "builds each bitcast as bitcasts.golden states"
      "bitcasts.golden" Goldens.bitcasts;
  ]

let substitute_goldens =
  [
    golden "substitute" "replaces each node as substitution.golden states"
      "substitution.golden" Goldens.substitution;
  ]

let groups =
  [
    const_group;
    literal_makers;
    value_group;
    is_literal_group;
    is_invalid_group;
    ccast_group;
    cconst_group;
    sink_group;
    kernel_nodes_group;
    group_group;
    is_group_group;
    broadcast_group;
    index_group;
    end_group;
    after_group;
    after_disjoint_group;
    is_disjoint_group;
    is_join_group;
    without_after_group;
    ins_group;
    range_group;
    loop_group;
    wmma_group;
    validity_group;
    device_group;
    on_disk_group;
    copy_to_device_group;
    size_src_group;
    base_group;
    storage_base_group;
    addrspace_group;
    whole_group;
    buf_uop_group;
    is_virtual_group;
    has_buffer_identity_group;
    needs_storage_group;
    unique_num_group;
    take_unique_group;
    new_buffer_group;
    set_group;
    variable_group;
    expr_group;
    const_factor_group;
    pop_const_group;
    body_group;
    src_without_body_group;
    opaque_call_bodies_group;
    is_inline_call_group;
  ]
  @ unbound_outputs_groups
  @ [ custom_function_group; call_group ]
  @ (elementwise :: elementwise_goldens)
  @ substitute_goldens
