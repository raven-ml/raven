(* Construction: the graphs the constructors build, elementwise arithmetic,
   movement, devices, storage, variables and calls. *)

open Windtrap
open Tolk
open Common

let str pp x = Format.asprintf "%a" pp x
let cpu = "CPU"
let param ?device shape slot dt = Shape.param ?device ~shape:(ints shape) slot dt

(* Graphs *)

let graph_goldens =
  group "graphs"
    (List.map
       (fun (name, build, minted) ->
         Golden.graph (name ^ ".golden") (fun () ->
             if minted then Goldens.minted (build ()) else build ()))
       Goldens.all)

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
      test "subtraction adds the negation" (fun () ->
          equal uop (Ops.add a (Ops.neg b)) (Ops.sub a b);
          equal uop (Ops.mul b (Ops.int (-1))) (Ops.neg b));
      test "greater is less with its operands swapped" (fun () ->
          equal uop (Ops.lt b a) (Ops.gt a b));
      test "at most, at least and equal negate a strict comparison" (fun () ->
          equal uop (Ops.logical_not (Ops.gt a b)) (Ops.le a b);
          equal uop (Ops.logical_not (Ops.lt a b)) (Ops.ge a b);
          equal uop (Ops.logical_not (Ops.ne a b)) (Ops.eq a b));
      test "logical_not compares the boolean cast with true" (fun () ->
          equal uop
            (Ops.ne (Ops.cast a Bool) (Ops.bool true))
            (Ops.logical_not a));
      test "negation and bitwise not of a boolean are logical not" (fun () ->
          let p = flag "p" in
          equal uop (Ops.logical_not p) (Ops.neg p);
          equal uop (Ops.logical_not p) (Ops.bitwise_not p));
      test "bitwise not xors with all ones" (fun () ->
          equal uop (Ops.bitwise_xor a (Ops.int (-1))) (Ops.bitwise_not a);
          let u = var ~dtype:Uint8 "u" 0 10 in
          equal uop (Ops.bitwise_xor u (Ops.int 255)) (Ops.bitwise_not u));
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
      test "consts takes the committed type of its literals" (fun () ->
          equal dtype Int32 (Ops.dtype (Shape.consts [ i 1; i 2 ]));
          equal dtype Int64
            (Ops.dtype
               (Shape.consts [ i 1; `Int (Bigint.shift_left Bigint.one 40) ]));
          equal dtype Float32 (Ops.dtype (Shape.consts [ i 1; f 2.5 ]));
          equal dtype Bool (Ops.dtype (Shape.consts [ `Bool true; `Bool false ])));
      test "const_like expands to the node's shape, in its type" (fun () ->
          let p = param [ 2; 3 ] 0 Float16 in
          let c = Shape.const_like p (i 1) in
          equal dtype Float16 (Ops.dtype c);
          equal shape (ints [ 2; 3 ]) (Shape.shape c);
          equal uop (Ops.int ~dtype:Float16 1) (Ops.base c));
    ]

(* Nodes of a kernel *)

let kernel_nodes =
  let a = var "a" 0 1 in
  group "kernel nodes"
    [
      test "stack rejects nothing to stack and differing shapes" (fun () ->
          rejects (fun () -> Shape.stack []);
          rejects (fun () ->
              Shape.stack [ param [ 2 ] 0 Float32; param [ 3 ] 1 Float32 ]));
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
      test "contract rejects a range that is not upcast" (fun () ->
          let r = Ops.range (Int 2) [ 0 ] in
          rejects (fun () -> Shape.contract Ops.O.(r + int 1) [ r ]));
      test "range defaults to a weak integer loop of weak role" (fun () ->
          let r = Ops.range (Int 4) [ 0 ] in
          equal dtype Weak_int (Ops.dtype r);
          is_true (Ops.axis_type r = Weak);
          equal uop (Ops.int 4) (Ops.nth r 0));
      test "a loop is unbounded and void" (fun () ->
          let l = Ops.loop 2 in
          equal dtype Void (Ops.dtype l);
          equal (list int) [ 2 ] (Ops.axis_id l);
          equal op Op.Noop (Ops.op (Ops.nth l 0)));
    ]

(* Movement *)

(* Up to three axes of up to four elements, some of them named in either
   order. *)
let gen_flip =
  Gen.(
    let* dims = list ~size:(int_range 1 3) (int_range 0 4) in
    let+ flags = list ~size:(constant (List.length dims)) bool
    and+ backwards = bool in
    let axes =
      List.filter_map Fun.id
        (List.mapi (fun a f -> if f then Some a else None) flags)
    in
    (dims, if backwards then List.rev axes else axes))

(* The row-major position in [dims] of the element that output position [i]
   holds once [axes] are reversed. *)
let flipped dims axes i =
  let rank = List.length dims in
  let rec coords a i acc =
    if a < 0 then acc
    else
      let d = List.nth dims a in
      let j = if List.mem a axes then d - 1 - (i mod d) else i mod d in
      coords (a - 1) (i / d) (j :: acc)
  in
  List.fold_left2 (fun acc j d -> (acc * d) + j) 0 (coords (rank - 1) i []) dims

let movement =
  let p = param [ 2; 3; 4 ] 0 Float32 in
  group "movement"
    [
      test "reshape infers a -1 and rejects two" (fun () ->
          equal shape
            (ints [ 4; 6 ])
            (Shape.shape (Shape.reshape p (ints [ 4; -1 ])));
          rejects (fun () -> Shape.reshape p (ints [ -1; -1 ])));
      test "reshape rejects a different number of elements" (fun () ->
          rejects (fun () -> Shape.reshape p (ints [ 5; 5 ])));
      test "cumalu past 512 elements runs in two stages" (fun () ->
          let sums = Shape.cumalu (Shape.arange ~dtype:Int32 600) 0 Op.Add in
          equal (list int)
            (List.init 600 (fun i -> i * (i + 1) / 2))
            (List.map
               (function `Int z -> Bigint.to_int z | _ -> -1)
               (Array.to_list (Tensors.eval sums))));
      test "arange rejects a step of 0 and a dtype that cannot hold it"
        (fun () ->
          rejects (fun () -> Shape.arange ~step:0 4);
          rejects (fun () -> Shape.arange ~dtype:Int8 200));
      test "rop rejects a repeated axis" (fun () ->
          rejects (fun () -> Shape.rop p Op.Add [ 1; 1 ]));
      test "permute rejects an order that is not a permutation" (fun () ->
          rejects (fun () -> Shape.permute p [ 0; 0; 1 ]);
          rejects (fun () -> Shape.permute p [ 0; 1 ]);
          rejects (fun () -> Shape.permute p [ 0; 1; 3 ]));
      test "mop rejects an empty pad or shrink of a node with axes" (fun () ->
          rejects (fun () -> Shape.mop p (Pad []));
          rejects (fun () -> Shape.mop p (Shrink []));
          let s = Ops.int 1 in
          is_true (Shape.mop s (Pad []) == s);
          is_true (Shape.mop s (Shrink []) == s));
      test "mop stores a shape argument as sources and reads it back" (fun () ->
          let r = Shape.mop p (Reshape (ints [ 6; 4 ])) in
          equal uops
            [ p; Ops.v ~src:[ Ops.int 6; Ops.int 4 ] Op.Stack ]
            (Ops.src r);
          equal shape
            (ints [ 6; 4 ])
            (match Shape.marg r with Reshape s -> s | _ -> fail "a reshape");
          equal shape (ints [ 6; 4 ]) (Shape.as_shape (Ops.nth r 1));
          equal shape (ints [ 3 ]) (Shape.as_shape (Ops.int 3)));
      test "as_shape of a node is the node" (fun () ->
          let n = weak_var "n" 1 4 in
          equal shape [ Sym n ] (Shape.as_shape n));
      test "marg reads each movement's argument and rejects other nodes"
        (fun () ->
          (match Shape.marg (Shape.permute p [ 2; 0; 1 ]) with
          | Permute o -> equal (list int) [ 2; 0; 1 ] o
          | _ -> fail "permute");
          (match
             Shape.marg (Shape.shrink p [ Some (Int 1, Int 2); None; None ])
           with
          | Shrink l ->
              equal
                (list (pair sint sint))
                [ (Int 1, Int 1); (Int 0, Int 3); (Int 0, Int 4) ]
                l
          | _ -> fail "shrink");
          (match Shape.marg (Shape.pad p [ Some (Int 1, Int 2); None; None ]) with
          | Pad l ->
              equal
                (list (pair sint sint))
                [ (Int 1, Int 5); (Int 0, Int 3); (Int 0, Int 4) ]
                l
          | _ -> fail "pad");
          rejects (fun () -> Shape.marg (Ops.int 1)));
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
      test "flip rejects an axis given twice, and is the node for no axis"
        (fun () ->
          rejects (fun () -> Shape.flip p [ 0; 0 ]);
          rejects (fun () -> Shape.flip (Shape.param 1 Float32) [ 0 ]);
          is_true (Shape.flip (Shape.param 1 Float32) [] == Shape.param 1 Float32);
          is_true (Shape.flip p [] == p));
      test "repeat refuses a negative repeat" (fun () ->
          rejects (fun () -> Shape.repeat (param [ 1 ] 1 Float32) [ -1 ]));
      test "a flip of a view of variable size reverses it at each size"
        (fun () ->
          let n = Ops.variable "n" (`Int Bigint.zero) (`Int (Bigint.of_int 4)) in
          let x = param [ 4 ] 1 Int32 in
          let flipped = Shape.flip (Shape.shrink x [ Some (Int 0, Sym n) ]) [ 0 ] in
          let memory : Dtype.value array =
            Array.init 4 (fun j -> `Int (Bigint.of_int j))
          in
          List.iter
            (fun k ->
              let bound =
                Ops.substitute ~calls:Skip ~pass:Fixed_point flipped
                  [ (n, Ops.int k) ]
              in
              equal ~msg:(string_of_int k) (list Dtypes.const)
                (List.init k (fun i -> `Int (Bigint.of_int (k - 1 - i))))
                (Array.to_list (Tensors.eval ~buffers:[ (1, memory) ] bound)))
            [ 0; 1; 2; 3; 4 ]);
      prop "a flip reverses each axis it names and keeps the others"
        gen_flip (fun (dims, axes) ->
          let n = List.fold_left ( * ) 1 dims in
          let flips_long = List.exists (fun a -> List.nth dims a > 1) axes in
          cover "an empty axis beside a flipped one"
            (List.mem 0 dims && flips_long);
          cover "an axis of one element is flipped"
            (List.exists (fun a -> List.nth dims a = 1) axes);
          cover "two axes are flipped" (List.length axes >= 2);
          let x = Shape.reshape (param [ n ] 1 Int32) (ints dims) in
          let memory : Dtype.value array =
            Array.init n (fun j -> `Int (Bigint.of_int j))
          in
          equal (list Dtypes.const)
            (List.init n (fun i ->
                 (memory.(flipped dims axes i) :> Dtype.const)))
            (Array.to_list
               (Tensors.eval ~buffers:[ (1, memory) ] (Shape.flip x axes))));
      test "shrink, pad and pad_to reject a list of another length" (fun () ->
          rejects (fun () -> Shape.shrink p [ None ]);
          rejects (fun () -> Shape.pad p [ None; None ]);
          rejects (fun () -> Shape.pad_to p [ None ]));
      test "pad_to to the same shape is the node, whatever the value" (fun () ->
          is_true (Shape.pad_to ~value:(f 1.) p [ None; None; None ] == p));
      test "squeeze leaves an axis of another size" (fun () ->
          is_true (Shape.squeeze ~axis:1 p == p));
      test "cat rejects shapes that differ off its axis" (fun () ->
          rejects (fun () -> Shape.cat p [ param [ 2; 4; 4 ] 1 Float32 ]));
      test "nbytes is the elements times their size" (fun () ->
          equal int 96 (Shape.nbytes p);
          equal int 2 (Shape.nbytes (Shape.param 1 Float16)));
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
      test "empty_like on another device takes the value's shape" (fun () ->
          let u =
            Shape.empty_like ~dtype:Int32 ~device:"CPU:2"
              (param ~device:cpu [ 8; 6 ] 1 Float32)
          in
          equal shape (ints [ 8; 6 ]) (Shape.shape u);
          equal (option device) (Some "CPU:2") (Ops.device u);
          is_true (Ops.has_buffer_identity u));
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
      test "placeholder rejects a scalar variable's address space" (fun () ->
          rejects (fun () ->
              Shape.placeholder ~slot:0 ~addrspace:Alu [ 4 ] Float32));
      test "storage rejects a weak type" (fun () ->
          rejects (fun () -> Ops.new_buffer cpu 4 Weak_float);
          rejects (fun () -> Shape.empty ~device:cpu (ints [ 4 ]) Weak_int);
          rejects (fun () -> Shape.param 0 Weak_int));
      test "placeholder commits a weak type" (fun () ->
          equal dtype Int32 (Ops.dtype (Shape.placeholder ~slot:0 [ 4 ] Weak_int)));
      test "a clone of a weak value commits its type" (fun () ->
          equal dtype Int32
            (Ops.dtype
               (Shape.clone ~device:cpu (Shape.expand (Ops.int 3) (ints [ 4 ])))));
      test "a stage is its own base and storage, without buffer identity"
        (fun () ->
          let s = Ops.bufferize (param ~device:cpu [ 2; 4 ] 0 Float32) [] in
          is_true (Ops.base s == s);
          is_true (Ops.buf_uop s == s);
          is_false (Ops.has_buffer_identity s);
          equal shape (ints [ 2; 4 ]) (Shape.shape s));
      test "clone rejects a disk, named in any case" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          rejects (fun () -> Shape.clone ~device:"DISK:/tmp/f" p);
          rejects (fun () -> Shape.clone ~device:"disk:0" p);
          is_true
            (String.equal "DISKS"
               (Option.get (Ops.device (Shape.clone ~device:"DISKS" p)))));
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
      test "param_like names the parameter it makes, or leaves it unnamed"
        (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          equal string "x" (Ops.expr (Shape.param_like ~name:"x" p 1));
          equal string "y" (Ops.expr (Shape.param_like ~name:"y" n 2));
          rejects (fun () -> Ops.expr (Shape.param_like n 2));
          rejects (fun () -> Ops.expr (Shape.param_like p 1)));
      test "bind binds a value within the range" (fun () ->
          let b = Shape.bind n (i 3) in
          is_true (Shape.is_bound_var b);
          is_true (Shape.is_variable b);
          equal (pair uop value) (n, i 3) (Shape.unbind b);
          ignore (Shape.bind n (i 1));
          ignore (Shape.bind n (i 8)));
      test
        "bind rejects a bound variable, a value out of range, and a value off \
         its multiple" (fun () ->
          rejects (fun () -> Shape.bind (Shape.bind n (i 3)) (i 4));
          rejects (fun () -> Shape.bind n (i 0));
          rejects (fun () -> Shape.bind n (i 9));
          rejects (fun () -> Shape.bind (weak_var ~multiple_of:4 "m" 0 16) (i 6));
          ignore (Shape.bind (weak_var ~multiple_of:4 "m" 0 16) (i 12));
          rejects (fun () -> Shape.bind (param [ 4 ] 0 Float32) (i 1)));
      test "unbound strips the value" (fun () ->
          is_true (Shape.unbound (Shape.bind n (i 3)) == n);
          is_true (Shape.unbound n == n));
      test "unbound rejects a node that is not a variable" (fun () ->
          rejects (fun () -> Shape.unbound (param [ 4 ] 0 Float32)));
      test "variables lists a scalar parameter without a range" (fun () ->
          let s = Shape.param ~addrspace:Alu 0 Int32 in
          equal uops [ s ] (Shape.variables Ops.O.(s + int 1)));
      test "unbind rejects a variable that is not bound" (fun () ->
          rejects (fun () -> Shape.unbind n));
      test "unbind_all unbinds each variable and lists its value" (fun () ->
          let m = weak_var "m" 0 4 in
          let e = Ops.O.(Shape.bind n (i 3) + Shape.bind m (i 2)) in
          let u, values = Shape.unbind_all e in
          equal uop Ops.O.(n + m) u;
          equal
            (slist (pair uop value) (fun (a, _) (b, _) -> Ops.compare a b))
            [ (n, i 3); (m, i 2) ]
            values);
      test "variables are sorted by name" (fun () ->
          let m = weak_var "m" 0 4 in
          equal uops [ m; n ] (Shape.variables Ops.O.(n + m));
          equal uops [ m; n ] (Shape.variables Ops.O.(Shape.bind n (i 3) + m)));
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
          let r = Ops.range (Int 4) [ 0 ] in
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
      test
        "call_with_outputs rejects output positions that do not ascend within \
         the arguments" (fun () ->
          let v = Ops.O.(arg + float 1.) in
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 1; 0 ] [ v; v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 0; 0 ] [ v; v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 5 ] [ v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 0; 1 ] [ v ] [ arg ]));
      test "a call compiled apart stages its arguments" (fun () ->
          let v = Ops.O.(arg + float 1.) in
          let out = Shape.call_with_output ~precompile:true v [ v ] in
          equal op Op.Stage (Ops.op (Ops.nth (Ops.nth out 1) 1)));
      test "custom_function names an external function and its type" (fun () ->
          let fn = Ops.custom_function "f" [ arg ] in
          equal op Op.Custom_function (Ops.op fn);
          equal Common.arg
            (Custom_function { name = "f"; dtype = Void })
            (Ops.arg fn);
          equal uops [ arg ] (Ops.src fn);
          equal dtype Int32
            (Ops.dtype (Ops.custom_function ~dtype:Int32 "f" [])));
    ]

let groups =
  [
    graph_goldens;
    elementwise;
    constants;
    kernel_nodes;
    movement;
    devices;
    storage;
    variables;
    calls;
  ]
