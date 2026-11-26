(* Construction: the graphs the constructors build, elementwise arithmetic,
   constants, kernel nodes, movement, devices, storage, variables and calls. *)

open Windtrap
open Tolk
open Common

let str pp x = Format.asprintf "%a" pp x
let cpu = "CPU"
let param ?device shape slot dt = Shape.param ?device ~shape:(ints shape) slot dt

(* Graphs *)

(* [golden value claim file build] is the group [value] › [claim] holding the
   test that [build] makes the graph golden [file]. *)
let golden value claim file build =
  group value [ group claim [ Golden.graph file build ] ]

let construction_goldens =
  [
    golden "const" "builds each constant as typed_constants.golden states"
      "typed_constants.golden" Goldens.typed_constants;
    golden "ccast" "builds each constant cast as ccast.golden states"
      "ccast.golden" Goldens.ccast;
    golden "kernel constructors"
      "end_, after, barrier, loop, range, backedge, index and group build \
       each node as kernel_nodes.golden states"
      "kernel_nodes.golden" Goldens.kernel_nodes;
    golden "ins" "builds each instruction as instructions.golden states"
      "instructions.golden" Goldens.instructions;
    golden "wmma" "builds each product as wmmas.golden states" "wmmas.golden"
      Goldens.wmmas;
    golden "get_idx and get_valid"
      "split each validity as validity.golden states" "validity.golden"
      Goldens.validity;
    golden "set" "stores through each pointer as sets.golden states"
      "sets.golden" Goldens.sets;
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
      "build each comparison as comparisons.golden states"
      "comparisons.golden" Goldens.comparisons;
    golden "the bitwise operations"
      "build each node as bitwise.golden states" "bitwise.golden"
      Goldens.bitwise;
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

(* Constants *)

let constants =
  group "constants"
    [
      test "a constant without a type is its weak literal" (fun () ->
          equal dtype Weak_int (Ops.dtype (Ops.int 3));
          equal dtype Weak_float (Ops.dtype (Ops.float 3.));
          equal dtype Bool (Ops.dtype (Ops.bool true));
          equal op Op.Const (Ops.op (Ops.int 3)));
      test
        "a constant of a type other than its literal's is a cast of the literal"
        (fun () ->
          let c = Ops.int ~dtype:Int32 3 in
          equal op Op.Cast (Ops.op c);
          equal uop (Ops.int 3) (Ops.nth c 0);
          equal uop (Ops.cast (Ops.float 2.) Float32) (Ops.int ~dtype:Float32 2);
          is_true (Ops.int ~dtype:Weak_int 3 == Ops.int 3));
      test "Invalid ignores the type" (fun () ->
          is_true (Ops.const ~dtype:Float32 `Invalid == Ops.invalid);
          equal dtype Bool (Ops.dtype Ops.invalid));
      test "value reads a constant through its cast, and rejects anything else"
        (fun () ->
          equal const (i 3) (Ops.value (Ops.int ~dtype:Int32 3));
          equal const (f 1.5) (Ops.value (Ops.float 1.5));
          equal const `Invalid (Ops.value Ops.invalid);
          rejects (fun () -> Ops.value (var "a" 0 1));
          rejects (fun () -> Ops.value (Ops.cast (var "a" 0 1) Float32)));
      test "is_invalid holds for the Invalid constant only" (fun () ->
          is_true (Ops.is_invalid Ops.invalid);
          is_false (Ops.is_invalid (Ops.bool false));
          is_false (Ops.is_invalid (Shape.valid (Ops.int 1) (flag "p"))));
      test "cconst casts even to the literal's own type" (fun () ->
          equal uop
            (Ops.v ~src:[ Ops.bool true ] ~arg:(Dtype Bool) Op.Cast)
            (Ops.cconst Bool (`Bool true)));
    ]

(* Nodes of a kernel *)

let kernel_nodes =
  let a = var "a" 0 1 in
  group "kernel nodes"
    [
      test "an index of a stack by a constant is the element" (fun () ->
          let s = Shape.stack [ a; var "b" 0 1 ] in
          is_true (Ops.index s [ Ops.int 0 ] == a);
          equal op Op.Index (Ops.op (Ops.index s [ Ops.int ~dtype:Int32 0 ])));
      test
        "an index of a stack by a negative constant counts from the end, as a \
         Python tuple" (fun () ->
          let b = var "b" 0 1 in
          let s = Shape.stack [ a; b ] in
          is_true (Ops.index s [ Ops.int (-1) ] == b);
          rejects (fun () -> Ops.index s [ Ops.int 2 ]);
          rejects (fun () -> Ops.index s [ Ops.int (-3) ]));
      test "an end of no ranges, and an after of nothing, are the node"
        (fun () ->
          is_true (Ops.end_ a [] == a);
          is_true (Ops.after a [] == a));
      test "without_after strips every after" (fun () ->
          let e = Ops.sink [] in
          equal uop a (Ops.without_after (Ops.after (Ops.after a [ e ]) [ e ]));
          is_true (Ops.without_after a == a));
      test "group of one node is the node" (fun () ->
          is_true (Ops.group [ a ] == a));
      test "sink keeps its sources in order" (fun () ->
          let b = var "b" 0 1 in
          equal uops [ b; a; b ] (Ops.src (Ops.sink [ b; a; b ])));
      test "nth is the source at a position, and rejects any other position"
        (fun () ->
          let b = var "b" 0 1 and c = var "c" 0 1 in
          let s = Ops.sink [ a; b; c ] in
          equal uops [ a; b; c ] (List.map (Ops.nth s) [ 0; 1; 2 ]);
          List.iter
            (fun i -> rejects (fun () -> Ops.nth s i))
            [ -1; 3; max_int; min_int ];
          rejects (fun () -> Ops.nth (Ops.sink []) 0));
      test "range defaults to a weak integer loop of weak role" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
          equal dtype Weak_int (Ops.dtype r);
          is_true (Ops.axis_type r = Weak);
          equal uop (Ops.int 4) (Ops.nth r 0));
      test "a loop is unbounded and void" (fun () ->
          let l = Ops.loop 2 in
          equal dtype Void (Ops.dtype l);
          equal (list int) [ 2 ] (Ops.axis_id l);
          equal op Op.Noop (Ops.op (Ops.nth l 0)));
      test "a group of one node is that node" (fun () ->
          equal uop a (Ops.group [ a ]));
    ]

(* Movement *)

let movement =
  let p = param [ 2; 3; 4 ] 0 Float32 in
  group "movement"
    [
      test "base strips movements, and storage_base bitcasts and afters"
        (fun () ->
          let flat = Ops.base p in
          equal op Op.Param (Ops.op flat);
          is_true
            (Ops.base (Shape.permute (Shape.reshape p (ints [ 6; 4 ])) [ 1; 0 ])
            == flat);
          let viewed =
            Ops.bitcast
              (Ops.after (param ~device:cpu [ 4 ] 2 Uint32) [ Ops.sink [] ])
              Int32
          in
          equal op Op.Param (Ops.op (Ops.storage_base viewed)));
    ]

(* Devices *)

let devices =
  group "devices"
    [
      test "device reads storage and copies, and sources otherwise" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          equal (option device) (Some cpu) (Ops.device p);
          equal (option device) (Some cpu) (Ops.device Ops.O.(p + float 1.));
          equal (option device) (Some "CUDA")
            (Ops.device (Ops.copy_to_device p "CUDA"));
          equal (option device) None (Ops.device (var "a" 0 1)));
      test "on_disk holds for a disk device" (fun () ->
          is_true
            (Ops.on_disk (param ~device:"DISK:/tmp/f" [ 4 ] 0 Uint8));
          is_false (Ops.on_disk (param ~device:cpu [ 4 ] 0 Uint8)));
      test "copy_to_device rejects a disk and a weak type" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          rejects (fun () -> Ops.copy_to_device p "DISK:/tmp/f");
          rejects (fun () -> Ops.copy_to_device p "disk");
          rejects (fun () -> Ops.copy_to_device p "disk:0");
          rejects (fun () -> Ops.copy_to_device (Ops.int 1) cpu));
      test "the device of a deep graph needs no deep recursion" (fun () ->
          let rec deepen n u =
            if n = 0 then u else deepen (n - 1) Ops.O.(u + u)
          in
          equal (option device) (Some cpu)
            (Ops.device (deepen 10_000 (Ops.new_buffer ~slot:0 cpu 1 Int8))));
    ]

(* Storage *)

let storage =
  group "storage"
    [
      test "addrspace of a stack is its sources' shared space" (fun () ->
          let local = Shape.placeholder ~slot:0 ~addrspace:Local [ 8 ] Float32 in
          let at n = Ops.index local [ Ops.int n ] in
          equal (option addr_space) (Some Local)
            (Ops.addrspace (Ops.v ~src:[ at 0; at 1 ] Op.Stack));
          equal (option addr_space) None
            (Ops.addrspace
               (Ops.v ~src:[ at 0; param [ 1 ] 1 Float32 ] Op.Stack)));
      test "addrspace reads storage, and passes through indexing and movement"
        (fun () ->
          let local = Shape.placeholder ~slot:0 ~addrspace:Local [ 8 ] Float32 in
          equal (option addr_space) (Some Local) (Ops.addrspace local);
          equal (option addr_space) (Some Local)
            (Ops.addrspace (Ops.index local [ Ops.int 0 ]));
          equal (option addr_space) (Some Global)
            (Ops.addrspace (param [ 2; 2 ] 1 Float32));
          equal (option addr_space) (Some Alu) (Ops.addrspace (var "a" 0 1));
          equal (option addr_space) (Some Alu)
            (Ops.addrspace (Ops.special (Int 4) "gidx0"));
          equal (option addr_space) (Some Alu)
            (Ops.addrspace (Ops.load (Ops.index local [ Ops.int 0 ]) []));
          equal (option addr_space) (Some Global)
            (Ops.addrspace (Ops.v ~arg:(Bytes "a") Op.Binary));
          equal (option addr_space) None
            (Ops.addrspace Ops.O.(local + Ops.int ~dtype:Float32 1)));
      test "buf_uop is the storage a node accesses" (fun () ->
          let p = param [ 4 ] 0 Float32 in
          is_true (Ops.buf_uop p == p);
          is_true (Ops.buf_uop (Ops.index p [ Ops.int 0 ]) == p);
          let s =
            Ops.contiguous Ops.O.(param ~device:cpu [ 4 ] 1 Float32 + float 1.)
          in
          is_true (Ops.buf_uop s == s));
      test "is_virtual holds without a device, or for a weak type" (fun () ->
          is_true (Ops.is_virtual (param [ 4 ] 0 Float32));
          is_false (Ops.is_virtual (param ~device:cpu [ 4 ] 0 Float32));
          is_true
            (Ops.is_virtual
               (Ops.cast (param ~device:cpu [ 4 ] 0 Float32) Weak_float)));
      test
        "has_buffer_identity sees storage through reshapes and on request \
         afters" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          is_true (Ops.has_buffer_identity p);
          is_true (Ops.has_buffer_identity (Shape.reshape p (ints [ 2; 2 ])));
          is_false
            (Ops.has_buffer_identity
               (Shape.permute (Shape.reshape p (ints [ 2; 2 ])) [ 1; 0 ]));
          let a = Ops.after p [ Ops.sink [] ] in
          is_false (Ops.has_buffer_identity a);
          is_true (Ops.has_buffer_identity ~after_ok:true a));
      test "needs_storage holds for a placed value without storage" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          is_false (Ops.needs_storage p);
          is_true (Ops.needs_storage Ops.O.(p + float 1.));
          is_true
            (Ops.needs_storage
               (Shape.alloc ~slot:3 ~device:cpu (ints [ 4 ]) Float32));
          is_false (Ops.needs_storage Ops.O.(param [ 4 ] 1 Float32 + float 1.)));
      test "unique_num never returns a number twice, from any domain" (fun () ->
          let draw () = List.init 500 (fun _ -> Ops.unique_num ()) in
          let all =
            List.concat
              (List.map Domain.join (List.init 4 (fun _ -> Domain.spawn draw)))
          in
          equal int (List.length all)
            (List.length (List.sort_uniq Int.compare all)));
      test "a buffer given a slot is no buffer made without one" (fun () ->
          (* The number the next fresh buffer would take, given by hand first. *)
          let slot = Ops.unique_num () + 1 in
          let given = Ops.new_buffer ~slot cpu 8 Float32 in
          let made = Ops.new_buffer cpu 8 Float32 in
          not_equal int (Ops.param_arg_of given).slot
            (Ops.param_arg_of made).slot);
      test "a buffer cannot be given the last number" (fun () ->
          rejects (fun () -> Ops.new_buffer ~slot:max_int cpu 8 Float32));
      test "a stage is its own base and storage, without buffer identity"
        (fun () ->
          let s = Ops.bufferize (param ~device:cpu [ 2; 4 ] 0 Float32) [] in
          is_true (Ops.base s == s);
          is_true (Ops.buf_uop s == s);
          is_false (Ops.has_buffer_identity s);
          equal shape (ints [ 2; 4 ]) (Shape.shape s));
      test "new_buffer takes the next slot without one" (fun () ->
          let slot u =
            match Ops.arg u with Param p -> p.slot | _ -> fail "a buffer"
          in
          not_equal int
            (slot (Ops.new_buffer cpu 4 Float32))
            (slot (Ops.new_buffer cpu 4 Float32)));
    ]

(* Variables *)

let variables =
  let n = weak_var "n" 1 8 in
  group "variables"
    [
      test "a variable is a named scalar with a range" (fun () ->
          is_true (Shape.is_variable n);
          is_false (Shape.is_bound_var n);
          equal string "n" (Ops.expr n);
          equal (pair value value) (int_bounds 1 8) (bounds n);
          is_false (Shape.is_variable (param [ 4 ] 0 Float32));
          is_false (Shape.is_variable (Shape.param 0 Int32)));
      test "expr names a named buffer, and rejects a node without a name"
        (fun () ->
          equal string "w"
            (Ops.expr (Shape.param ~name:"w" ~shape:(ints [ 4 ]) 0 Float32));
          rejects (fun () -> Ops.expr (param [ 4 ] 0 Float32)));
    ]

(* Calls *)

let calls =
  let body = Ops.sink [] in
  let arg = param ~device:cpu [ 4 ] 0 Float32 in
  group "calls"
    [
      test "body is a call's first source, and rejects anything else" (fun () ->
          equal uop body (Ops.body (Ops.call body [ arg ]));
          rejects (fun () -> Ops.body body));
      test "src_without_body leaves out a call's body only" (fun () ->
          equal uops [ arg ] (Ops.src_without_body (Ops.call body [ arg ]));
          equal uops [ arg ] (Ops.src_without_body (Ops.sink [ arg ])));
      test "opaque_call_bodies is the operations a call body can have"
        (fun () ->
          equal (list op)
            Op.[ Program; Linear; Sink; Store; Custom_function ]
            (Op.Set.to_list Ops.opaque_call_bodies));
      test "call rejects a body that computes a value" (fun () ->
          rejects (fun () -> Ops.call (Ops.int 1) [ arg ]);
          rejects (fun () -> Ops.call Ops.O.(arg + float 1.) []));
      test "call rejects a range leaking out of its body" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
          rejects (fun () ->
              Ops.call
                (Ops.sink
                   [
                     Ops.store (Ops.index arg [ r ])
                       (Ops.float ~dtype:Float32 1.);
                   ])
                []));
      test "is_inline_call holds for a plain sink that is not compiled apart"
        (fun () ->
          is_true (Ops.is_inline_call (Ops.call body []));
          is_false (Ops.is_inline_call (Ops.call ~precompile:true body []));
          is_false
            (Ops.is_inline_call
               (Ops.call (Ops.sink ~kernel:(Ops.kernel_info ()) []) []));
          is_false (Ops.is_inline_call body));
      test "a call with outputs has them unbound until they are resolved"
        (fun () ->
          let out = Shape.call_with_output Ops.O.(arg + float 1.) [ arg ] in
          let c = Ops.nth out 1 in
          is_true (Ops.has_unbound_outputs c);
          equal uops [ out ] (Ops.unbound_outputs c);
          is_false (Ops.has_unbound_outputs (Ops.call body [ arg ]));
          is_false (Ops.has_unbound_outputs arg));
      test "custom_function names an external function and its type" (fun () ->
          let fn = Ops.custom_function "f" [ arg ] in
          equal op Op.Custom_function (Ops.op fn);
          equal Common.arg
            (Custom_function { name = "f"; dtype = Void; fmt = false })
            (Ops.arg fn);
          equal uops [ arg ] (Ops.src fn);
          equal dtype Int32
            (Ops.dtype (Ops.custom_function ~dtype:Int32 "f" [])));
    ]

let groups =
  (elementwise :: elementwise_goldens)
  @ [ constants; kernel_nodes ]
  @ construction_goldens
  @ [ movement; devices; storage; variables; calls ]
  @ substitute_goldens
