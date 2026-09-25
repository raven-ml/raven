(* Smoke tests for the public tolk.uop API. *)

open Windtrap
open Tolk_uop

let storage_view ~src ~offset ~size ~dtype =
  let module U = Tolk_uop.Uop in
  let module D = Tolk_uop.Dtype in
  let offset = U.alu_binary ~op:Tolk_uop.Ops.Mul ~lhs:offset
      ~rhs:(U.const_int (D.itemsize (U.dtype src))) in
  let bytes = U.bitcast ~src ~dtype:D.int8 in
  U.bitcast ~dtype ~src:(U.shrink ~src:bytes ~offset
      ~size:(U.const_int (size * D.itemsize dtype)))

let string_option =
  let pp fmt = function
    | None -> Format.pp_print_string fmt "None"
    | Some s -> Format.fprintf fmt "Some %S" s
  in
  Testable.make ~pp ~equal:(Option.equal String.equal)

let op_testable = Testable.make ~pp:Ops.pp ~equal:Ops.equal

let launch_value_testable =
  let pp fmt = function
    | Uop.Launch_value_int n -> Format.fprintf fmt "Launch_value_int %d" n
    | Uop.Launch_value_float f -> Format.fprintf fmt "Launch_value_float %g" f
  in
  let equal a b =
    match a, b with
    | Uop.Launch_value_int a, Uop.Launch_value_int b -> a = b
    | Uop.Launch_value_float a, Uop.Launch_value_float b ->
        Float.equal a b
    | _ -> false
  in
  Testable.make ~pp ~equal

let contains haystack needle =
  let n = String.length haystack and m = String.length needle in
  let rec loop i =
    i + m <= n
    && (String.sub haystack i m = needle || loop (i + 1))
  in
  m = 0 || loop 0

let shape_ints u =
  List.map
    (fun dim ->
      match Uop.const_int_value dim with
      | Some n -> n
      | None -> fail "expected concrete shape dimension")
    (Uop.shape u)

let equal_bounds ~msg u expected =
  equal (pair int int) ~msg expected (Bound.to_int (Uop.vmin u), Bound.to_int (Uop.vmax u))

let with_env name value f =
  let old = Sys.getenv_opt name in
  Unix.putenv name value;
  try
    let result = f () in
    (match old with
     | Some old_value -> Unix.putenv name old_value
     | None -> Unix.putenv name "");
    result
  with exn ->
    (match old with
     | Some old_value -> Unix.putenv name old_value
     | None -> Unix.putenv name "");
    raise exn

(* Construction + accessors *)

let ops_access () =
  let _ = Ops.Add in
  let _ = Dtype.int32 in
  is_true ~msg:"Ops.Add is ALU" (Ops.Group.is_alu Ops.Add)

let ops_tinygrad_order () =
  let expected =
    [
      "SPECIAL";
      "BUFFER";
      "ALLOC";
      "NOOP";
      "REWRITE_ERROR";
      "PARAM";
      "CALL";
      "PROGRAM";
      "LINEAR";
      "SOURCE";
      "BINARY";
      "SINK";
      "AFTER";
      "GROUP";
      "STACK";
      "GETADDR";
      "INDEX";
      "SHRINK";
      "LOAD";
      "STORE";
      "WMMA";
      "CAST";
      "BITCAST";
      "EXP2";
      "LOG2";
      "SIN";
      "SQRT";
      "RECIPROCAL";
      "NEG";
      "TRUNC";
      "ADD";
      "MUL";
      "SHL";
      "SHR";
      "CDIV";
      "MAX";
      "CMOD";
      "CMPLT";
      "CMPNE";
      "CMPEQ";
      "XOR";
      "OR";
      "AND";
      "THREEFRY";
      "SUB";
      "FDIV";
      "POW";
      "FLOORDIV";
      "FLOORMOD";
      "WHERE";
      "MULACC";
      "BARRIER";
      "RANGE";
      "IF";
      "END";
      "ENDIF";
      "BACKEDGE";
      "WAIT";
      "CONST";
      "CUSTOM";
      "CUSTOMI";
      "INS";
      "CONTIGUOUS_BACKWARD";
      "DETACH";
      "STAGE";
      "COPY";
      "MSELECT";
      "MSTACK";
      "CUSTOM_FUNCTION";
      "RESHAPE";
      "PERMUTE";
      "EXPAND";
      "PAD";
      "FLIP";
      "UNSHARD";
      "REDUCE";
      "ALLREDUCE";
      "PYLITERAL";
    ]
  in
  is_true ~msg:"Ops.Group.all matches tinygrad order"
    (List.map Ops.name Ops.Group.all = expected)

let group_algebra () =
  let open Ops.Group in
  let agrees name group pred =
    is_true ~msg:(name ^ " predicate agrees with group")
      (List.for_all (fun op -> pred op = mem op group) all)
  in
  is_true ~msg:"mem finds binary op" (mem Ops.Add binary);
  is_true ~msg:"mem rejects absent op" (not (mem Ops.Range binary));
  let merged = union [ Ops.Add; Ops.Mul ] [ Ops.Mul; Ops.Max ] in
  is_true ~msg:"union preserves unique order"
    (merged = [ Ops.Add; Ops.Mul; Ops.Max ]);
  is_true ~msg:"without removes comparisons"
    (without binary comparison
     =
     [
       Ops.Add;
       Ops.Mul;
       Ops.Shl;
       Ops.Shr;
       Ops.Cdiv;
       Ops.Max;
       Ops.Cmod;
       Ops.Xor;
       Ops.Or;
       Ops.And;
       Ops.Threefry;
       Ops.Sub;
       Ops.Fdiv;
       Ops.Pow;
       Ops.Floordiv;
       Ops.Floormod;
     ]);
  is_true ~msg:"tinygrad reduce ops are add/mul/max"
    (reduce = [ Ops.Add; Ops.Mul; Ops.Max ]);
  is_true ~msg:"Defines include bound and unbound storage"
    (defines = [ Ops.Buffer; Ops.Alloc; Ops.Param ]);
  is_true ~msg:"Irreducible includes Param and Getaddr"
    (irreducible = [ Ops.Special; Ops.Param; Ops.Getaddr; Ops.Range; Ops.Const ]);
  is_true ~msg:"broadcastable excludes Group"
    (not (is_broadcastable Ops.Group));
  is_true ~msg:"Cdiv and Cmod are binary"
    (is_binary Ops.Cdiv && is_binary Ops.Cmod);
  is_true ~msg:"floor div/mod are binary"
    (is_binary Ops.Floordiv && is_binary Ops.Floormod);
  agrees "unary" unary is_unary;
  agrees "binary" binary is_binary;
  agrees "ternary" ternary is_ternary;
  agrees "alu" alu is_alu;
  agrees "broadcastable" broadcastable is_broadcastable;
  agrees "elementwise" elementwise is_elementwise;
  agrees "defines" defines is_define;
  agrees "irreducible" irreducible is_irreducible;
  agrees "movement" movement is_movement;
  agrees "commutative" commutative is_commutative;
  agrees "associative" associative is_associative;
  agrees "idempotent" idempotent is_idempotent;
  agrees "reduce" reduce is_reduce;
  agrees "comparison" comparison is_comparison

let hashcons_identity () =
  let sum = Uop.alu_binary ~op:Ops.Add ~lhs:(Uop.const_int 1) ~rhs:(Uop.const_int 2) in
  let sum' = Uop.alu_binary ~op:Ops.Add ~lhs:(Uop.const_int 1) ~rhs:(Uop.const_int 2) in
  is_true ~msg:"structural identity" (sum == sum')

let add_has_two_srcs () =
  let sum = Uop.alu_binary ~op:Ops.Add ~lhs:(Uop.const_int 1) ~rhs:(Uop.const_int 2) in
  is_true ~msg:"op is Add" (Uop.op sum = Ops.Add);
  is_true ~msg:"two srcs" (Array.length (Uop.src sum) = 2)

let infix_builds_mul () =
  let open Uop.O in
  let e = (Uop.const_int 3 + Uop.const_int 4) * int_ 2 in
  is_true ~msg:"outer op is Mul" (Uop.op e = Ops.Mul)

(* Validity masks combine through these folds, and the valid simplifier
   splits conjunctions on [And]. *)
let bool_folds_are_logical () =
  let flag name = Uop.variable ~name ~min_val:0 ~max_val:1 ~dtype:Dtype.bool () in
  let a = flag "a" and b = flag "b" in
  equal op_testable Ops.And (Uop.op (Uop.uprod [ a; b ]));
  equal op_testable Ops.Or (Uop.op (Uop.usum [ a; b ]));
  let x = Uop.variable ~name:"x" ~min_val:0 ~max_val:8 () in
  equal op_testable Ops.Mul (Uop.op (Uop.uprod [ x; x ]));
  equal op_testable Ops.Add (Uop.op (Uop.usum [ x; x ]))

let arithmetic_helpers_tinygrad_parity () =
  let open Uop.O in
  let x = Uop.variable ~name:"x" ~min_val:0 ~max_val:64 () in
  let y = Uop.variable ~name:"y" ~min_val:0 ~max_val:64 () in
  let z = Uop.variable ~name:"z" ~min_val:0 ~max_val:64 () in
  is_true ~msg:"/ builds tinygrad true division"
    (Uop.op (x / y) = Ops.Fdiv);
  is_true ~msg:"// builds tinygrad floor division"
    (Uop.op (x // y) = Ops.Floordiv);
  is_true ~msg:"mod builds tinygrad floor modulo"
    (Uop.op (x mod y) = Ops.Floormod);
  is_true ~msg:"cdiv builds truncating division explicitly"
    (Uop.op (cdiv x y) = Ops.Cdiv);
  is_true ~msg:"cmod builds truncating modulo explicitly"
    (Uop.op (cmod x y) = Ops.Cmod);
  is_true ~msg:"named floor helpers build floor ops"
    (Uop.op (floordiv x y) = Ops.Floordiv
     && Uop.op (floormod x y) = Ops.Floormod);
  let typed = Uop.const (Const.int Dtype.int32 8) in
  equal int 8 (Uop.const_factor typed);
  let quotient = Option.get (Uop.divides typed 2) in
  equal (option int) (Some 4) (Uop.const_int_value quotient);
  is_true (Dtype.equal (Uop.dtype quotient) Dtype.int32);
  let stack = Uop.stack [ x * int_ 2; y * int_ 4 ] in
  equal int ~msg:"empty STACK const_factor follows math.gcd()"
    0 (Uop.const_factor (Uop.stack []));
  equal int ~msg:"STACK const_factor is gcd of lanes"
    2 (Uop.const_factor stack);
  (match Uop.divides stack 2 with
   | Some quotient ->
       is_true ~msg:"STACK divides reconstructs a STACK"
         (Uop.op quotient = Ops.Stack);
       is_true ~msg:"STACK divides preserves vector dtype"
         (Dtype.equal (Uop.dtype quotient) (Uop.dtype stack));
       equal int ~msg:"STACK quotient keeps lane gcd"
         1 (Uop.const_factor quotient)
   | None -> fail "STACK should divide by shared constant factor");
  (match Uop.divide_exact stack (int_ 2) with
   | Some quotient ->
       is_true ~msg:"divide_exact over STACK delegates constant divisors"
         (Uop.op quotient = Ops.Stack);
       equal int ~msg:"exact STACK quotient const_factor"
         1 (Uop.const_factor quotient)
   | None -> fail "STACK should divide exactly by constant factor");
  let divisor_stack = Uop.stack [ int_ 2; int_ 4 ] in
  is_true ~msg:"STACK / STACK exact division stays unsupported like tinygrad"
    (Uop.divide_exact stack divisor_stack = None);
  (match Uop.divide_exact ((x * y) + (x * z)) x with
   | Some quotient ->
       is_true ~msg:"shared factor divides each additive term"
         (Uop.equal quotient (y + z))
   | None -> fail "shared symbolic factor should divide sum exactly");
  let common = Uop.gcd [ x * y; x * z ] in
  is_true ~msg:"gcd keeps common symbolic product factor"
    (Uop.equal common x)

let param_arg_symbolic_constructor () =
  let v = Uop.variable ~name:"n" ~min_val:2 ~max_val:8 () in
  is_true ~msg:"variable is scalar Buffer" (Uop.is_variable v);
  (match Uop.arg v with
   | Uop.Arg.Param_arg { slot; name; vmin_vmax; addrspace; _ } ->
       is_true ~msg:"symbolic slot" (slot = -1);
       is_true ~msg:"symbolic name" (name = Some "n");
       is_true ~msg:"symbolic bounds" (vmin_vmax = Some (Bound.int 2, Bound.int 8));
       is_true ~msg:"symbolic addrspace" (addrspace = Dtype.Alu)
   | _ -> is_true ~msg:"Param carries Param_arg" false);
  (match Uop.as_buffer v with
   | Some { buffer = { name; vmin_vmax; _ }; shape } ->
       is_true ~msg:"param view name" (name = Some "n");
       is_true ~msg:"param view bounds" (vmin_vmax = Some (Bound.int 2, Bound.int 8));
       is_true ~msg:"variable has scalar shape child"
         (Uop.op shape = Ops.Stack && Dtype.equal (Uop.dtype shape) Dtype.void)
   | None -> is_true ~msg:"param view is present" false);
  is_true ~msg:"vmin reads Param_arg" ((Bound.to_int (Uop.vmin v)) = 2);
  is_true ~msg:"vmax reads Param_arg" ((Bound.to_int (Uop.vmax v)) = 8)

let bind_requires_concrete_value () =
  let var = Uop.variable ~name:"n" ~min_val:0 ~max_val:4 () in
  let value = Uop.const_int 3 in
  let bound = Uop.bind ~var ~value in
  is_true ~msg:"binding effect" (Uop.is_bound_var bound);
  (match Uop.as_bind bound with
   | Some { var = v; value = got } ->
       is_true ~msg:"bind keeps symbolic param" (v == var);
       is_true ~msg:"bind keeps concrete value" (got == value)
   | None -> is_true ~msg:"bind view" false);
  let out_of_bounds_rejected =
    try
      ignore (Uop.bind ~var ~value:(Uop.const_int 9));
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"out-of-range bind value rejected" out_of_bounds_rejected;
  let var = Uop.variable ~name:"aligned" ~min_val:0 ~max_val:16 ~multiple_of:4 () in
  raises (Invalid_argument "Uop.bind: value violates variable divisor")
    (fun () -> ignore (Uop.bind ~var ~value:(Uop.const_int 6)));
  let bound = Uop.bind ~var ~value:(Uop.const_int 12) in
  equal int64 12L (snd (Uop.unbind bound));
  equal int 4 (Uop.const_factor var);
  is_true ~msg:"variable has no runtime allocation"
    (match Uop.as_buffer var with
     | Some { buffer = { buffer = None; size = None; _ }; _ } -> true
     | _ -> false)

let integer_bounds_parity () =
  let empty_range =
    Uop.range ~size:(Uop.const_int 0) ~axis:0 ~kind:Axis_type.Weak ()
  in
  equal_bounds ~msg:"RANGE(0) is an empty interval" empty_range (0, -1);
  let empty_special =
    Uop.special ~name:"lidx0" ~size:(Uop.const_int 0) ()
  in
  equal_bounds ~msg:"SPECIAL(0) is an empty interval" empty_special (0, -1);
  let symbolic_size =
    Uop.variable ~name:"n" ~min_val:0 ~max_val:4 ()
  in
  let bounded_special =
    Uop.special ~name:"lidx1" ~size:symbolic_size ()
  in
  equal_bounds ~msg:"SPECIAL upper bound follows size.vmax - 1"
    bounded_special (0, 3);
  let two = Uop.const_int 2 in
  let neg_two = Uop.const_int (-2) in
  let mixed = Uop.variable ~name:"x" ~min_val:(-5) ~max_val:5 () in
  equal_bounds ~msg:"FLOORDIV uses Python floor division"
    (Uop.alu_binary ~op:Ops.Floordiv ~lhs:mixed ~rhs:two)
    (-3, 2);
  equal_bounds ~msg:"FLOORDIV handles negative divisors"
    (Uop.alu_binary ~op:Ops.Floordiv ~lhs:mixed ~rhs:neg_two)
    (-3, 2);
  equal_bounds ~msg:"FLOORDIV over empty numerator is zero"
    (Uop.alu_binary ~op:Ops.Floordiv ~lhs:empty_range ~rhs:two)
    (0, 0);
  let same_floor = Uop.variable ~name:"same" ~min_val:4 ~max_val:5 () in
  equal_bounds ~msg:"FLOORMOD constant positive divisor same quotient"
    (Uop.alu_binary ~op:Ops.Floormod ~lhs:same_floor ~rhs:(Uop.const_int 3))
    (1, 2);
  let crosses_floor = Uop.variable ~name:"cross" ~min_val:4 ~max_val:8 () in
  equal_bounds ~msg:"FLOORMOD constant positive divisor crossing quotient"
    (Uop.alu_binary ~op:Ops.Floormod ~lhs:crosses_floor ~rhs:(Uop.const_int 3))
    (0, 2);
  equal_bounds ~msg:"FLOORMOD constant negative divisor same quotient"
    (Uop.alu_binary ~op:Ops.Floormod ~lhs:same_floor ~rhs:(Uop.const_int (-3)))
    (-2, -1);
  equal_bounds ~msg:"FLOORMOD positive divisor range"
    (Uop.alu_binary ~op:Ops.Floormod ~lhs:mixed
       ~rhs:(Uop.variable ~name:"d" ~min_val:3 ~max_val:5 ()))
    (0, 4);
  equal_bounds ~msg:"FLOORMOD negative divisor range"
    (Uop.alu_binary ~op:Ops.Floormod ~lhs:mixed
       ~rhs:(Uop.variable ~name:"nd" ~min_val:(-5) ~max_val:(-3) ()))
    (-4, 0);
  equal_bounds ~msg:"FLOORMOD over empty numerator is zero"
    (Uop.alu_binary ~op:Ops.Floormod ~lhs:empty_range ~rhs:two)
    (0, 0);
  let exact_bounds u lo hi =
    is_true (Bound.equal (Uop.vmin u) (`Int lo));
    is_true (Bound.equal (Uop.vmax u) (`Int hi))
  in
  let signed_max = Z.of_int64 Int64.max_int in
  let signed_min = Z.of_int64 Int64.min_int in
  let unsigned_max = Z.pred (Z.shift_left Z.one 64) in
  exact_bounds (Uop.const (Const.int64 Dtype.int64 Int64.max_int)) signed_max signed_max;
  exact_bounds (Uop.const (Const.int64 Dtype.int64 Int64.min_int)) signed_min signed_min;
  exact_bounds (Uop.const (Const.int64 Dtype.uint64 Int64.minus_one)) unsigned_max unsigned_max;
  exact_bounds (Uop.param ~slot:7 ~dtype:Dtype.uint64 ()) Z.zero unsigned_max;
  let wrapping_const =
    Uop.const
      (Const.int64 Dtype.weakint (Int64.add Int64.min_int 5L))
  in
  let overflow_rejected =
    try
      ignore
        (Uop.bind
           ~var:(Uop.variable ~name:"small" ~min_val:0 ~max_val:8 ())
           ~value:wrapping_const);
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"bind rejects int64 values outside native bounds"
    overflow_rejected

(* Fixed-width integer arithmetic wraps: an interval that would leave the
   dtype's range holds any of its values, one inside it stays exact. *)
let wrapping_integer_bounds () =
  let c dt n = Uop.const (Const.int dt n) in
  let u8 = Uop.param ~slot:0 ~dtype:Dtype.uint8 () in
  equal_bounds ~msg:"uint8 x - 1"
    (Uop.alu_binary ~op:Ops.Add ~lhs:u8 ~rhs:(c Dtype.uint8 (-1))) (0, 255);
  equal_bounds ~msg:"uint8 x * 2"
    (Uop.alu_binary ~op:Ops.Mul ~lhs:u8 ~rhs:(c Dtype.uint8 2)) (0, 255);
  let low = Uop.alu_binary ~op:Ops.And ~lhs:u8 ~rhs:(c Dtype.uint8 15) in
  equal_bounds ~msg:"uint8 (x & 15) + 1"
    (Uop.alu_binary ~op:Ops.Add ~lhs:low ~rhs:(c Dtype.uint8 1)) (1, 16);
  let i8 = Uop.param ~slot:1 ~dtype:Dtype.int8 () in
  equal_bounds ~msg:"int8 x + 1"
    (Uop.alu_binary ~op:Ops.Add ~lhs:i8 ~rhs:(c Dtype.int8 1)) (-128, 127)

let flat_storage_parameters () =
  let n = Uop.variable ~name:"extent" ~min_val:1 ~max_val:8 () in
  let dims = Uop.stack [Uop.const_int 3; n] in
  let view = Uop.param ~slot:4 ~dtype:Dtype.float32 ~shape:dims () in
  let base = Uop.buf_uop view in
  equal int 0 (Array.length (Uop.src base));
  equal int 24 (Uop.max_numel base);
  is_true (List.equal Uop.equal [Uop.const_int 3; n] (Uop.shape view));
  (match Uop.as_param base with
   | Some { param; _ } -> equal (option int) (Some 24) param.size
   | None -> fail "expected flat parameter");
  let concrete = Uop.param ~slot:5 ~dtype:Dtype.int32
      ~shape:(Uop.const (Const.int Dtype.int32 4)) () in
  is_true ~msg:"concrete strong dimensions need no symbolic shrink"
    (Uop.op concrete = Ops.Param);
  let scalar = Uop.param ~slot:0 ~dtype:Dtype.int32 () in
  equal int 0 (Array.length (Uop.src scalar));
  is_true (Uop.shape scalar = []);
  let image = Uop.param ~slot:1 ~dtype:Dtype.float32 ~image:(2, 3) () in
  equal int 0 (Array.length (Uop.src image));
  equal (list int) [2; 3; 4] (Uop.max_shape image);
  match Uop.as_param image with
  | Some { param; _ } -> equal (option int) (Some 24) param.size
  | None -> fail "expected image parameter"

let max_numel_checks_host_range () =
  let dim = Uop.const_int (1 lsl 32) in
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Uop.param ~slot:0 ~dtype:Dtype.float32
        ~shape:(Uop.stack [ dim; dim ]) ())

let allocations_preserve_shape_and_address_space () =
  let scalar = Uop.alloc ~dtype:Dtype.weakint ~shape:(Uop.stack []) () in
  equal (list int) [] (Uop.max_shape scalar);
  is_true (Dtype.equal (Uop.dtype scalar) Dtype.int32);
  let p = Option.get (Uop.Arg.as_param_arg (Uop.arg (Uop.storage_base scalar))) in
  equal (option int) (Some 1) p.size;
  let n = Uop.variable ~name:"allocation_extent" ~min_val:2 ~max_val:8 () in
  List.iter (fun addrspace ->
      let make () = Uop.alloc ~dtype:Dtype.float32 ~shape:(Uop.stack [n; Uop.const_int 3])
          ~addrspace () in
      let a = make () and b = make () in
      is_false ~msg:"anonymous allocations have distinct storage" (Uop.storage_base a == Uop.storage_base b);
      is_true (Uop.op (Uop.storage_base a) = Ops.Alloc);
      is_true (Uop.addrspace a = Some addrspace);
      is_true (List.equal Uop.equal [n; Uop.const_int 3] (Uop.shape a));
      equal int 24 (Uop.max_numel a);
      equal (list int) [8; 3] (Uop.max_shape (Uop.alloc_like a ~addrspace ())))
    [Dtype.Global; Dtype.Local; Dtype.Reg];
  List.iter (fun addrspace ->
      raises_match (function Invalid_argument _ -> true | _ -> false)
        (fun () -> Uop.alloc ~dtype:Dtype.float32 ~shape:(Uop.const_int 4)
            ~addrspace ~device:(Uop.Single "CPU") ())) [Dtype.Local; Dtype.Reg]

let placeholder_checks_shape_product () =
  List.iter (fun addrspace ->
      raises_match (function Invalid_argument _ -> true | _ -> false)
        (fun () -> Uop.placeholder ~shape:[1 lsl 32; 1 lsl 32]
            ~dtype:Dtype.float32 ~slot:0 ~addrspace ());
      let empty = Uop.placeholder ~shape:[1 lsl 32; 1 lsl 32; 0]
          ~dtype:Dtype.float32 ~slot:0 ~addrspace () in
      equal int 0 (Uop.max_numel empty))
    [Dtype.Global; Dtype.Local; Dtype.Reg]

let max_numel_handles_zero_after_large_dimensions () =
  let huge = Uop.const (Const.integer Dtype.weakint (Z.shift_left Z.one 100)) in
  let buffer = Uop.param ~slot:0 ~dtype:Dtype.float32
      ~shape:(Uop.stack [ huge; Uop.const_int 0 ]) () in
  equal int 0 (Uop.max_numel buffer);
  equal int 0 (Uop.max_shard_numel buffer)

let movement_dimensions_do_not_wrap () =
  let base = Uop.param ~slot:0 ~dtype:Dtype.float32
      ~shape:(Uop.const_int 2) () in
  let invalid_shape f =
    raises_match (function Invalid_argument _ -> true | _ -> false) f in
  invalid_shape (fun () -> Uop.shape
      (Uop.shrink ~src:base ~offset:(Uop.const_int max_int)
         ~size:(Uop.const_int 2)));
  invalid_shape (fun () -> Uop.shape
      (Uop.pad ~src:base ~offset:(Uop.const_int max_int)
         ~size:(Uop.const_int 0)));
  let empty = Uop.param ~slot:1 ~dtype:Dtype.float32
      ~shape:(Uop.const_int 0) () in
  invalid_shape (fun () -> Uop.shape
      (Uop.reshape ~src:empty
         ~shape:(Uop.stack [ Uop.const_int (1 lsl 31); Uop.const_int (1 lsl 32) ])))

let bitcast_dimensions_remain_exact_until_host_conversion () =
  let base = Uop.param ~slot:0 ~dtype:Dtype.uint64
      ~shape:(Uop.const_int max_int) () in
  let bytes = Uop.bitcast ~src:base ~dtype:Dtype.uint8 in
  let expected = Z.mul (Z.of_int max_int) (Z.of_int 8) in
  (match Uop.shape bytes with
   | [ dim ] -> is_true (Bound.equal (Uop.vmax dim) (`Int expected))
   | _ -> fail "expected one exact byte dimension");
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Uop.max_shape bytes);
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Uop.max_numel bytes)

let exact_symbolic_bounds () =
  let huge = Z.shift_left Z.one 200 in
  let value n = Uop.const (Const.integer Dtype.weakint n) in
  let exact u n =
    is_true (Bound.equal (Uop.vmin u) (`Int n));
    is_true (Bound.equal (Uop.vmax u) (`Int n))
  in
  exact Uop.O.(value huge - value Z.(pred huge)) Z.one;
  exact Uop.O.(value huge * value huge) Z.(mul huge huge);
  exact (Uop.contiguous ~src:(value huge) ~force:true ()) huge;
  let shifted = Uop.alu_binary ~op:Ops.Shl ~lhs:(value huge) ~rhs:(Uop.const_int 100) in
  exact shifted Z.(shift_left one 300);
  let v = Uop.param ~slot:(-1) ~dtype:Dtype.weakint ~name:"wide"
      ~vmin_vmax:(`Int huge, `Int Z.(add huge (of_int 7)))
      ~addrspace:Dtype.Alu ~shape:(Uop.stack []) () in
  let small = Uop.O.(v - value huge) in
  (match Symbolic.parse_valid Uop.O.(value huge < v) with
   | Some (subject, false, lo) ->
       is_true (Uop.equal subject v);
       is_true (Bound.equal lo (`Int (Z.succ huge)))
   | _ -> fail "expected exact lower-bound clause");
  equal_bounds ~msg:"cancellation retains a tight interval" small (0, 7);
  is_true (Uop.resolve ~default:false Uop.O.(small < Uop.const_int 8));
  let above_float = value (Z.of_string "9007199254740993") in
  let rounded_float = Uop.const (Const.float Dtype.float64 9007199254740992.) in
  let comparison = Uop.alu_binary ~op:Ops.Cmplt ~lhs:rounded_float ~rhs:above_float in
  exact comparison Z.one;
  let nan = Uop.const (Const.float Dtype.float32 Float.nan) in
  is_true (Bound.equal (Uop.vmin nan) (`Float neg_infinity));
  is_true (Bound.equal (Uop.vmax nan) (`Float infinity));
  let fractional = Uop.const (Const.float Dtype.float64 (-3.75)) in
  exact (Uop.cast ~src:fractional ~dtype:Dtype.int32) (Z.of_int (-3));
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Bound.to_int (`Int huge))

let cast_bounds_parity () =
  let fits = Uop.variable ~name:"fits" ~min_val:5 ~max_val:10 () in
  equal_bounds ~msg:"CAST to unsigned keeps exact bounds when the source fits"
    (Uop.cast ~src:fits ~dtype:Dtype.uint8) (5, 10);
  let maybe_negative =
    Uop.variable ~name:"maybe_negative" ~min_val:(-1) ~max_val:10 ()
  in
  equal_bounds ~msg:"CAST to unsigned of a negative source can wrap"
    (Uop.cast ~src:maybe_negative ~dtype:Dtype.uint8) (0, 255);
  let too_large = Uop.variable ~name:"too_large" ~min_val:250 ~max_val:260 () in
  equal_bounds ~msg:"CAST to unsigned of a too-large source can wrap"
    (Uop.cast ~src:too_large ~dtype:Dtype.uint8) (0, 255);
  let wide = Uop.variable ~name:"wide" ~min_val:(-300) ~max_val:300 () in
  equal_bounds ~msg:"CAST to a signed int clamps to the destination window"
    (Uop.cast ~src:wide ~dtype:Dtype.int8) (-128, 127);
  equal_bounds ~msg:"CAST to bool says nothing"
    (Uop.cast ~src:fits ~dtype:Dtype.bool) (0, 1);
  let round_trip =
    Uop.cast ~src:(Uop.cast ~src:fits ~dtype:Dtype.float32) ~dtype:Dtype.int32
  in
  equal_bounds ~msg:"CAST to int carries a float source's bounds through"
    round_trip (5, 10)

let stack_stage_slice_constructors () =
  let a = Uop.const_int 1 and b = Uop.const_int 2 in
  let stacked = Uop.stack [ a; b ] in
  is_true ~msg:"stack uses Stack op" (Uop.op stacked = Ops.Stack);
  let opts : Uop.stage_opts =
    { device = None; addrspace = Dtype.Global; removable = false }
  in
  let staged = Uop.stage ~src:a ~ranges:[] ~opts in
  is_true ~msg:"stage uses Stage op" (Uop.op staged = Ops.Stage);
  is_true ~msg:"stage inherits source dtype"
    (Dtype.equal (Uop.dtype staged) (Uop.dtype a))

let uop_constructor_parity_shortcuts () =
  let a = Uop.const_int 1 and b = Uop.const_int 2 in
  let stacked = Uop.stack [ a; b ] in
  let casted = Uop.cast ~src:stacked ~dtype:Dtype.float32 in
  is_true ~msg:"stack cast keeps scalar lane dtype"
    (Dtype.equal (Uop.dtype casted) Dtype.float32);
  equal (list int) ~msg:"stack cast keeps vector shape" [ 2 ]
    (List.map (fun dim -> Bound.to_int (Uop.vmax dim)) (Uop.shape casted));
  is_true ~msg:"cast to same dtype returns source"
    (Uop.cast ~src:stacked ~dtype:Dtype.weakint == stacked);
  is_true ~msg:"bitcast to same dtype returns source"
    (Uop.bitcast ~src:stacked ~dtype:(Uop.dtype stacked) == stacked);
  let buf = Uop.buffer ~slot:0 ~dtype:Dtype.int32 () in
  is_true ~msg:"buffer bitcast to same dtype returns source"
    (Uop.bitcast ~src:buf ~dtype:(Uop.dtype buf) == buf);
  is_true ~msg:"INDEX of stack with one const lane returns lane"
    (Uop.index ~ptr:stacked ~idxs:[ Uop.const_int 1 ] () == b);
  is_true ~msg:"committed lane indexes survive construction"
    (Uop.op (Uop.index ~ptr:stacked ~idxs:[ Uop.cconst (Const.int Dtype.weakint 1) Dtype.int32 ] ()) = Ops.Index);
  is_true ~msg:"negative bare stack index follows the reference tuple lookup"
    (Uop.index ~ptr:stacked ~idxs:[ Uop.const_int (-1) ] () == b);
  List.iter (fun i ->
      raises (Invalid_argument "Uop.index: stack index out of bounds")
        (fun () -> Uop.index ~ptr:stacked ~idxs:[ Uop.const_int i ] ())) [ -3; 2 ];
  equal string ~msg:"committed constants render their value"
    "32" (Tolk_uop.Render.expr_to_string ~simplify:false
      (Uop.cconst (Const.int Dtype.weakint 32) Dtype.int32));
  let v = Uop.variable ~name:"render_width" ~min_val:0 ~max_val:8 () in
  equal string ~msg:"nonconstant casts retain their width"
    "(int)(render_width)" (Tolk_uop.Render.expr_to_string ~simplify:false
      (Uop.cast ~src:v ~dtype:Dtype.int32));
  is_true ~msg:"empty end returns value"
    (Uop.end_ ~value:a ~ranges:[] == a);
  let contiguous = Uop.contiguous ~src:stacked () in
  is_true ~msg:"deviceless contiguous returns source"
    (contiguous == stacked);
  let buffer = Uop.buffer ~slot:0 ~dtype:Dtype.int32 () in
  is_true ~msg:"contiguous buffer identity is useless"
    (Uop.contiguous ~src:buffer () == buffer)

let const_scalar_payload_constructors () =
  let open Uop in
  let scalar = const_of_dtype Dtype.int32 (Const_scalar (`Int 2L)) in
  is_true ~msg:"scalar const carries an explicit width" (op scalar = Ops.Cast);
  is_true ~msg:"scalar const keeps its dtype"
    (Dtype.equal (dtype scalar) Dtype.int32);
  (match as_const scalar with
   | Some c ->
       is_true ~msg:"scalar const keeps value" (Const.view c = Const.Int (Z.of_int 2))
   | _ -> is_true ~msg:"scalar const payload" false);
  let coerced = const_of_dtype Dtype.float32 (Const_scalar (`Int 2L)) in
  is_true ~msg:"scalar const coerced to requested dtype"
    (Dtype.equal (dtype coerced) Dtype.float32);
  (match as_const coerced with
   | Some c ->
       is_true ~msg:"int payload is coerced to float"
         (match Const.view c with Const.Float 2.0 -> true | _ -> false)
   | _ -> is_true ~msg:"coerced payload" false);
  let nan = const_of_dtype Dtype.float32 (Const_scalar (`Float Float.nan)) in
  (match as_const nan with
   | Some c ->
       is_true ~msg:"nan payload is canonical NaN"
         (match Const.view c with Const.Float f -> Float.is_nan f | _ -> false);
       is_true ~msg:"nan const equals itself" (Const.equal c c)
   | _ -> is_true ~msg:"nan payload" false);
  let neg_zero = const_of_dtype Dtype.float32 (Const_scalar (`Float (-0.0))) in
  let pos_zero = const_of_dtype Dtype.float32 (Const_scalar (`Float 0.0)) in
  (match as_const neg_zero, as_const pos_zero with
   | Some a, Some b ->
       is_true ~msg:"-0.0 and 0.0 stay distinct"
         (not (Const.equal a b))
   | _ -> is_true ~msg:"zero payloads" false);
  let invalid = const_of_dtype Dtype.weakint Const_invalid in
  is_true ~msg:"invalid const is a Const" (op invalid = Ops.Const);
  (match as_const invalid with
   | Some c ->
       is_true ~msg:"invalid payload is Invalid"
         (match Const.view c with Const.Invalid -> true | _ -> false)
   | _ -> is_true ~msg:"invalid payload" false)

let call_constructor_parity () =
  let info =
    {
      Uop.grad_fxn = None;
      name = Some (Uop.Label "call");
      precompile = false;
      precompile_backward = false;
      dtype = Dtype.void;
      aux = None;
    }
  in
  let arg = Uop.const_int 2 in
  raises (Invalid_argument "Uop.call: value-producing bodies require call_with_outputs")
    (fun () -> ignore (Uop.call ~body:(Uop.const_int 1) ~args:[arg] ~info));
  let outputs = Uop.call_with_outputs ~values:[Uop.const (Const.int Dtype.int32 1)]
      ~args:[arg] ~info () in
  (match outputs with
   | [output] ->
       is_true ~msg:"returned value sequences an explicit allocation"
         (Uop.op output = Ops.After && Uop.op (Uop.storage_base output) = Ops.Alloc);
       (match Uop.as_call (Uop.src output).(1) with
        | Some {body; args = [input; out]; _} ->
            is_true ~msg:"body stores its result" (Uop.op body = Ops.Sink);
            is_true ~msg:"input stays in its slot" (input == arg);
            is_true ~msg:"output is an explicit argument" (Uop.storage_base out == Uop.storage_base output)
        | _ -> fail "expected input and output call arguments")
   | _ -> fail "expected one output");
  let sink = Uop.sink [] in
  let opaque_call = Uop.call ~body:sink ~args:[ arg ] ~info in
  is_true ~msg:"sink body call stays Call"
    (Ops.equal (Uop.op opaque_call) Ops.Call);
  let range =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Weak ()
  in
  let leaked_range_rejected =
    try
      ignore (Uop.call ~body:range ~args:[] ~info);
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"call rejects leaking ranges" leaked_range_rejected

let property_helpers_parity () =
  let shape2 = Uop.stack [ Uop.const_int 2; Uop.const_int 4 ] in
  let param =
    Uop.param ~slot:0 ~dtype:Dtype.int32 ~shape:shape2
      ~axis:0 ~device:(Uop.Multi [ "CPU"; "GPU" ]) ()
  in
  is_true ~msg:"Sharded parameter carries an UNSHARD axis" (Uop.axis param = Some 0);
  equal (list int) ~msg:"Sharded parameter retains its logical shape"
    [ 2; 4 ] (shape_ints param);
  let multi = Uop.unshard ~src:param ~axes:[0] () in
  is_true ~msg:"Multi axis comes from arg" (Uop.axis multi = Some 0);
  equal (list int) ~msg:"Multi shape expands sharded axis"
    [ 4; 4 ] (shape_ints multi);
  equal (list int) ~msg:"Multi shard_shape divides sharded axis"
    [ 2; 4 ]
    (List.map
       (fun dim ->
         match Uop.const_int_value dim with
         | Some n -> n
         | None -> fail "expected concrete shard dimension")
       (Uop.shard_shape multi));
  equal (list int) ~msg:"Multi max_shard_shape uses shard dimensions"
    [ 2; 4 ] (Uop.max_shard_shape multi);
  equal int 8 (Uop.max_shard_numel multi);
  equal (list (pair int int)) ~msg:"Multi bounds follow shard size"
    [ (0, 2); (2, 4) ]
    (List.map
       (fun (lo, hi) ->
         match Uop.const_int_value lo, Uop.const_int_value hi with
         | Some lo, Some hi -> lo, hi
         | _ -> fail "expected concrete bounds")
       (Uop.bounds multi));
  let copied = Uop.copy ~src:multi ~device:(Uop.Single "CPU") () in
  is_true ~msg:"Copy clears axis" (Uop.axis copied = None);
  let var =
    Uop.variable ~name:"n" ~min_val:0 ~max_val:8 ()
  in
  let bound = Uop.bind ~var ~value:(Uop.const_int 3) in
  is_true ~msg:"scalar binding has no shard axis" (Uop.axis bound = None);
  is_true ~msg:"Param addrspace comes from ParamArg"
    (Uop.addrspace var = Some Dtype.Alu);
  let buffer =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:(Uop.const_int 8) ~addrspace:Dtype.Local ()
  in
  is_true ~msg:"Buffer addrspace comes from ParamArg"
    (Uop.addrspace buffer = Some Dtype.Local);
  let special =
    Uop.special ~name:"idx0" ~size:(Uop.const_int 4) ()
  in
  is_true ~msg:"Special is ALU addrspace"
    (Uop.addrspace special = Some Dtype.Alu);
  let opts : Uop.stage_opts =
    { device = Some (Uop.Single "CPU"); addrspace = Dtype.Local;
      removable = false }
  in
  let staged = Uop.stage ~src:param ~ranges:[] ~opts in
  is_true ~msg:"Stage axis inherits from source" (Uop.axis staged = Some 0);
  equal (list int) ~msg:"Stage shape appends source shape"
    [ 2; 4 ] (shape_ints staged);
  is_true ~msg:"Stage is its own base" (Uop.base staged == staged);
  is_true ~msg:"Stage is not a buffer identity"
    (not (Uop.has_buffer_identity staged));
  let reshaped = Uop.reshape ~src:multi ~shape:(Uop.const_int 16) in
  is_true ~msg:"Reshape remaps shard axis by prefix product"
    (Uop.axis reshaped = Some 0);
  equal (list int) ~msg:"Reshape shape comes from shape arg"
    [ 16 ] (shape_ints reshaped);
  (match Uop.marg reshaped with
   | Uop.Marg_shape dims ->
       equal (list int) ~msg:"Reshape marg is target shape"
         [ 16 ]
         (List.map
            (fun dim ->
              match Uop.const_int_value dim with
              | Some n -> n
              | None -> fail "expected concrete reshape marg")
            dims)
   | _ -> fail "expected Marg_shape for Reshape");
  is_true ~msg:"Base walks through movement and stops at Multi"
    (Uop.base reshaped == multi);
  let permed = Uop.permute ~src:multi ~order:[ 1; 0 ] in
  is_true ~msg:"Permute remaps shard axis" (Uop.axis permed = Some 1);
  equal (list int) ~msg:"Permute shape follows order"
    [ 4; 4 ] (shape_ints permed);
  (match Uop.marg permed with
   | Uop.Marg_permute order ->
       equal (list int) ~msg:"Permute marg is axis order" [ 1; 0 ] order
   | _ -> fail "expected Marg_permute for Permute");
  let flipped = Uop.flip ~src:multi ~dims:[ true; false ] in
  (match Uop.marg flipped with
   | Uop.Marg_flip dims ->
       equal (list bool) ~msg:"Flip marg is axis flags" [ true; false ] dims
   | _ -> fail "expected Marg_flip for Flip");
  let padded =
    Uop.pad ~src:multi
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 6; Uop.const_int 4 ])
  in
  equal (list int) ~msg:"Pad shape is output size"
    [ 6; 4 ] (shape_ints padded);
  (match Uop.marg padded with
   | Uop.Marg_bounds bounds ->
       equal (list (pair int int)) ~msg:"Pad marg is offset/size bounds"
         [ (1, 6); (0, 4) ]
         (List.map
            (fun (offset, size) ->
              match Uop.const_int_value offset, Uop.const_int_value size with
              | Some offset, Some size -> offset, size
              | _ -> fail "expected concrete pad marg")
            bounds)
   | _ -> fail "expected Marg_bounds for Pad");
  let kept =
    Uop.shrink ~src:multi
      ~offset:(Uop.stack [ Uop.const_int 0; Uop.const_int 1 ])
      ~size:(Uop.stack [ Uop.const_int 4; Uop.const_int 2 ])
  in
  is_true ~msg:"Shrink keeps axis when shard dim is full"
    (Uop.axis kept = Some 0);
  equal (list int) ~msg:"Shrink shape is slice size"
    [ 4; 2 ] (shape_ints kept);
  (match Uop.marg kept with
   | Uop.Marg_bounds bounds ->
       equal (list (pair int int)) ~msg:"Shrink marg is offset/size bounds"
         [ (0, 4); (1, 2) ]
         (List.map
            (fun (offset, size) ->
              match Uop.const_int_value offset, Uop.const_int_value size with
              | Some offset, Some size -> offset, size
              | _ -> fail "expected concrete shrink marg")
            bounds)
   | _ -> fail "expected Marg_bounds for Shrink");
  let sliced_axis =
    Uop.shrink ~src:multi
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 2; Uop.const_int 4 ])
  in
  is_true ~msg:"Shrink clears axis when shard dim is sliced"
    (Uop.axis sliced_axis = None);
  let reduced = Uop.reduce_axis ~src:multi ~op:Ops.Add ~axes:[ 0 ] in
  is_true ~msg:"Reduce clears reduced shard axis"
    (Uop.axis reduced = None);
  equal (list int) ~msg:"Reduce shape drops reduced dims"
    [ 4 ] (shape_ints reduced);
  let sliced =
    storage_view ~src:buffer ~offset:(Uop.const_int 0) ~size:4
      ~dtype:Dtype.int32
  in
  equal (list int) ~msg:"storage view shape is its size"
    [ 4 ] (shape_ints sliced);
  is_true ~msg:"storage view retains its buffer" (Uop.storage_base sliced == buffer);
  is_false ~msg:"movement views are not standalone storage identities"
    (Uop.has_buffer_identity sliced);
  is_true ~msg:"storage view buf_uop resolves through source"
    (Uop.buf_uop sliced == buffer);
  is_true ~msg:"Stage buf_uop stops at stage"
    (Uop.buf_uop staged == staged);
  let info =
    {
      Uop.grad_fxn = None;
      name = Some (Uop.Label "shape_fn");
      precompile = false;
      precompile_backward = false;
      dtype = Dtype.void;
      aux = None;
    }
  in
  let formal_dim =
    Uop.param ~slot:0 ~dtype:Dtype.weakint ~vmin_vmax:(Bound.int (1), Bound.int (8))
      ~name:"n" ~addrspace:Dtype.Alu ()
  in
  let body_value =
    Uop.expand ~src:(Uop.const (Const.float Dtype.float32 1.)) ~dims:formal_dim
  in
  let actual_dim = Uop.const_int 5 in
  let projected = List.hd (Uop.call_with_outputs ~values:[body_value]
      ~args:[actual_dim] ~info ()) in
  (match Uop.shape projected with
   | [ dim ] ->
       is_true
         ~msg:"call output shape substitutes formal Param by call arg"
         (dim == actual_dim)
   | _ -> fail "expected one substituted call output dimension");
  let shaped_const =
    Uop.const_of_dtype ~shape:shape2 Dtype.float32
      (Const_scalar (`Int 1L))
  in
  is_true ~msg:"Shaped const expands non-scalar shape"
    (Uop.op shaped_const = Ops.Expand);
  equal (list int) ~msg:"Shaped const has target shape"
    [ 2; 4 ] (shape_ints shaped_const)

let reduce_layouts () =
  let shaped =
    Uop.buffer ~slot:0 ~dtype:Dtype.float32
      ~shape:
        (Uop.stack [ Uop.const_int 2; Uop.const_int 3; Uop.const_int 4 ])
      ()
  in
  (* Reducing the leading axes needs no size-one drop, so the tensor reduce is
     a bare REDUCE whose reduced axes are permuted to the front and counted. *)
  let tensor_reduce = Uop.reduce_axis ~src:shaped ~op:Ops.Add ~axes:[ 0; 1 ] in
  is_true ~msg:"tensor reduce is a bare REDUCE"
    (Uop.op tensor_reduce = Ops.Reduce);
  (match Uop.as_reduce tensor_reduce with
   | Some { ranges; op; num_axes; _ } ->
       is_true ~msg:"tensor reduce has no lowered ranges" (ranges = []);
       is_true ~msg:"tensor reduce arg op" (Ops.equal op Ops.Add);
       is_true ~msg:"tensor reduce counts the reduced axes" (num_axes = 2)
   | None -> is_true ~msg:"tensor reduce view" false);
  equal (list int) ~msg:"tensor reduce drops reduced dims"
    [ 4 ] (shape_ints tensor_reduce);
  (* A size-one reduced axis is dropped by reshape rather than reduced, so no
     REDUCE node survives. *)
  let with_unit =
    Uop.buffer ~slot:1 ~dtype:Dtype.float32
      ~shape:
        (Uop.stack [ Uop.const_int 2; Uop.const_int 1; Uop.const_int 4 ])
      ()
  in
  let unit_reduce = Uop.reduce_axis ~src:with_unit ~op:Ops.Add ~axes:[ 1 ] in
  is_true ~msg:"size-one reduce drops to a reshape"
    (Uop.op unit_reduce = Ops.Reshape && Uop.as_reduce unit_reduce = None);
  equal (list int) ~msg:"size-one reduce keeps remaining dims"
    [ 2; 4 ] (shape_ints unit_reduce);
  is_true ~msg:"empty-axis tensor reduce is passthrough"
    (Uop.reduce_axis ~src:shaped ~op:Ops.Add ~axes:[] == shaped);
  let range =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Reduce ()
  in
  let lowered =
    Uop.reduce ~src:shaped ~ranges:[ range ] ~op:Ops.Add
  in
  (match Uop.as_reduce lowered with
   | Some { src; ranges; op; num_axes } ->
       is_true ~msg:"lowered reduce keeps body source" (src == shaped);
       is_true ~msg:"lowered reduce exposes ranges" (ranges = [ range ]);
       is_true ~msg:"lowered reduce arg op" (Ops.equal op Ops.Add);
       is_true ~msg:"lowered reduce has zero axis count" (num_axes = 0)
   | None -> is_true ~msg:"lowered reduce view" false);
  (match Uop.arg lowered with
   | Uop.Arg.Reduce_arg { op; num_axes } ->
       is_true ~msg:"lowered reduce carries Reduce_arg"
         (Ops.equal op Ops.Add && num_axes = 0)
   | _ -> is_true ~msg:"lowered reduce payload" false)

let binary_and_getaddr_dtypes () =
  let bin = Uop.binary "code" in
  is_true ~msg:"BINARY is uint8" (Dtype.equal (Uop.dtype bin) Dtype.uint8);
  equal (list int) ~msg:"BINARY has one shape dim per byte" [ 4 ]
    (shape_ints bin);
  let addr = Uop.getaddr ~src:(Uop.const_int 1) () in
  is_true ~msg:"GETADDR is uint64"
    (Dtype.equal (Uop.dtype addr) Dtype.uint64)

let void_and_value_op_shapes () =
  let a = Uop.const_int 1 in
  is_true ~msg:"CUSTOM effect has no shape"
    (Uop.shape_opt (Uop.custom ~fmt:"nop" ~args:[]) = None);
  is_true ~msg:"GROUP has no shape"
    (Uop.shape_opt (Uop.group [ a; Uop.const_int 2 ]) = None);
  is_true ~msg:"void INS has no shape"
    (Uop.shape_opt (Uop.ins ~mnemonic:"nop" ~operands:[] ()) = None);
  (match
     Uop.shape_opt
       (Uop.ins ~mnemonic:"mov" ~operands:[ a ] ~dtype:Dtype.int32 ())
   with
   | Some [] -> ()
   | _ -> fail "typed INS should have scalar shape");
  let vec = Uop.stack [ Uop.const_int 1; Uop.const_int 2 ] in
  let inlined = Uop.custom_inline ~fmt:"x" ~args:[ vec ] ~dtype:Dtype.int32 in
  is_true ~msg:"CUSTOM_INLINE is Customi" (Uop.op inlined = Ops.Customi);
  equal (list int) ~msg:"CUSTOM_INLINE shape follows its operands" [ 2 ]
    (shape_ints inlined)

let stack_prepends_leading_dim () =
  let vec =
    Uop.stack [ Uop.const_int 5; Uop.const_int 6; Uop.const_int 7 ]
  in
  equal (list int) ~msg:"STACK prepends a lane-count dimension" [ 3 ]
    (shape_ints vec);
  let matrix =
    Uop.stack
      [ vec; Uop.stack [ Uop.const_int 8; Uop.const_int 9; Uop.const_int 10 ] ]
  in
  equal (list int) ~msg:"nested STACK prepends the outer lane count" [ 2; 3 ]
    (shape_ints matrix)

let prepend_expand () =
  let base =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32
      ~shape:(Uop.stack [ Uop.const_int 4; Uop.const_int 5 ]) ()
  in
  let expanded = Uop.expand ~src:base ~dims:(Uop.stack [ Uop.const_int 3 ]) in
  is_true ~msg:"EXPAND is an Expand op" (Uop.op expanded = Ops.Expand);
  equal (list int) ~msg:"EXPAND prepends leading dims" [ 3; 4; 5 ]
    (shape_ints expanded);
  is_true ~msg:"empty EXPAND dims is identity"
    (Uop.expand ~src:base ~dims:(Uop.stack []) == base)


let bitcast_size_change () =
  let bytes3 =
    Uop.buffer ~slot:0 ~dtype:Dtype.int8
      ~shape:(Uop.stack [ Uop.const_int 3 ]) ()
  in
  let raised =
    try
      ignore (Uop.shape (Uop.bitcast ~src:bytes3 ~dtype:Dtype.int32));
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"bitcast with an indivisible trailing dim raises" raised;
  let bytes4 =
    Uop.buffer ~slot:1 ~dtype:Dtype.int8
      ~shape:(Uop.stack [ Uop.const_int 4 ]) ()
  in
  let widened = Uop.bitcast ~src:bytes4 ~dtype:Dtype.int32 in
  equal (list int) ~msg:"bitcast rescales the trailing dim to the wider itemsize"
    [ 1 ] (shape_ints widened)

let child_ops_reports_child_op_set () =
  let node =
    Uop.sink
      [
        Uop.const_int 1;
        Uop.const_int 2;
        Uop.variable ~name:"x" ~min_val:0 ~max_val:4 ();
      ]
  in
  let ops = Uop.child_ops node in
  is_true ~msg:"child_ops dedups child ops to a set"
    (List.length ops = 2
    && List.exists (Ops.equal Ops.Const) ops
    && List.exists (Ops.equal Ops.Buffer) ops);
  is_true ~msg:"child_ops is stable across calls" (Uop.child_ops node = ops)

let backward_slice_tracks_shared_dependencies () =
  let leaf = Uop.param ~slot:97 ~dtype:Dtype.int32 () in
  let branch = Uop.alu_binary ~op:Ops.Add ~lhs:leaf ~rhs:(Uop.const_int 3) in
  let root = Uop.sink [branch; leaf; branch] in
  let nodes = Uop.backward_slice root in
  equal (list int) (List.map Uop.tag (Uop.toposort root) |> List.filter ((<>) (Uop.tag root)))
    (List.map Uop.tag nodes);
  is_false ~msg:"a slice excludes its own root" (Uop.in_backward_slice root root);
  is_true ~msg:"shared inputs remain reachable" (Uop.in_backward_slice leaf root);
  let changed = Uop.replace root ~src:[| Uop.const_int 8 |] () in
  is_false ~msg:"replacement has an independent slice" (Uop.in_backward_slice leaf changed);
  is_true ~msg:"querying another root preserves the original slice"
    (Uop.in_backward_slice leaf root)

let property_caches_release_nodes () =
  let weak = Stdlib.Weak.create 4 in
  let[@inline never] populate () =
    let r = Uop.range ~axis:9127 ~kind:Axis_type.Weak ~size:(Uop.const_int 9) () in
    let cond = Uop.O.(r < Uop.const_int 4) in
    ignore (Uop.with_metadata [ { name = "lifetime"; backward = false } ] r);
    ignore (Uop.child_ops cond);
    ignore (Uop.device_of cond);
    ignore (Uop.addrspace cond);
    ignore (Uop.ranges r);
    ignore (Uop.bool_slice_mem cond cond);
    ignore (Uop.vmin r);
    ignore (Uop.shape r);
    ignore (Uop.backward_slice cond);
    ignore (Uop.in_backward_slice r cond);
    let ended = Uop.end_ ~value:cond ~ranges:[r] in
    let barrier = Uop.barrier ~srcs:[ended; ended] () in
    ignore (Uop.ranges barrier);
    Stdlib.Weak.set weak 0 (Some r);
    Stdlib.Weak.set weak 1 (Some cond);
    Stdlib.Weak.set weak 2 (Some ended);
    Stdlib.Weak.set weak 3 (Some barrier)
  in
  populate ();
  Gc.full_major ();
  Gc.full_major ();
  is_false ~msg:"cached range is collectible, even when its ranges include itself"
    (Stdlib.Weak.check weak 0);
  is_false ~msg:"cached condition is collectible" (Stdlib.Weak.check weak 1);
  is_false ~msg:"cached ending dependency is collectible" (Stdlib.Weak.check weak 2);
  is_false ~msg:"cached barrier is collectible" (Stdlib.Weak.check weak 3)

let exec_alu_folds_and_absorbs () =
  let c n = Const.int Dtype.int32 n in
  (match Uop.exec_alu Ops.Add Dtype.int32 [ c 2; c 3 ] with
   | Some r -> is_true ~msg:"Add folds constants" (Const.view r = Const.Int (Z.of_int 5))
   | None -> is_true ~msg:"Add folds constants" false);
  (match
     Uop.exec_alu Ops.Add Dtype.int32 [ c 2; Const.invalid ]
   with
   | Some r ->
       is_true ~msg:"an Invalid operand absorbs the fold"
         (match Const.view r with Const.Invalid -> true | _ -> false)
   | None -> is_true ~msg:"Invalid absorbs the fold" false);
  let byte n = Const.int Dtype.uint8 n in
  (match Uop.exec_alu Ops.Add Dtype.uint8 [ byte 255; byte 1 ] with
   | Some r ->
       is_true ~msg:"an add wraps to the dtype width"
         (Const.view r = Const.Int (Z.of_int 0))
   | None -> is_true ~msg:"an add folds" false)

let exec_alu_exact_scalars () =
  let check name op dtype args expected =
    let actual = Option.map Const.to_string (Uop.exec_alu op dtype args) in
    equal ~msg:name (option string) (Some expected) actual
  in
  let i = Const.int64 Dtype.int64 in
  check "preserves the high signed bits" Ops.Add Dtype.int64
    [ i 4611686018427387904L; i 0L ] "4611686018427387904:i64";
  check "compares signed int64 without narrowing" Ops.Cmplt Dtype.bool
    [ i Int64.max_int; i 0L ] "false:bool";
  check "wraps at the committed width" Ops.Add Dtype.int64
    [ i Int64.max_int; i 1L ] "-9223372036854775808:i64";
  check "compares unsigned int64 mathematically" Ops.Cmplt Dtype.bool
    [ Const.int64 Dtype.uint64 Int64.minus_one; Const.int Dtype.uint64 0 ]
    "false:bool";
  check "weak and strong scalar equality uses values" Ops.Cmpeq Dtype.bool
    [ Const.int Dtype.int32 1; Const.int Dtype.weakint 1 ] "true:bool";
  check "zero to a negative power" Ops.Pow Dtype.float64
    [ Const.float Dtype.float64 0.; Const.float Dtype.float64 (-1.) ] "inf:f64"

let exec_alu_float_division () =
  let fold x y =
    match Uop.exec_alu Ops.Fdiv Dtype.float64
      [ Const.float Dtype.float64 x; Const.float Dtype.float64 y ] with
    | Some c -> (match Const.view c with Const.Float f -> f | _ -> fail "expected float")
    | None -> fail "expected division to fold"
  in
  is_true ~msg:"zero divided by zero is NaN" (Float.is_nan (fold 0. 0.));
  equal float_exact Float.neg_infinity (fold 1. (-0.));
  is_true ~msg:"NaN divided by zero remains NaN" (Float.is_nan (fold Float.nan 0.))

let scalar_float_to_weak_integer () =
  let c = Const.of_scalar Dtype.weakint (`Float (2. ** 80.)) in
  equal string "1208925819614629174706176:weakint" (Const.to_string c)

let scalar_width_boundaries () =
  let check op dt args expected =
    equal (option string) (Some expected)
      (Option.map Const.to_string (Uop.exec_alu op dt args))
  in
  let u n = Const.int64 Dtype.uint64 n in
  check Ops.Cdiv Dtype.uint64 [ u Int64.minus_one; u 3L ] "6148914691236517205:u64";
  check Ops.Shr Dtype.uint64 [ u Int64.minus_one; u 63L ] "1:u64";
  check Ops.Cmpeq Dtype.bool
    [ Const.int64 Dtype.weakint 9007199254740993L;
      Const.float Dtype.float64 9007199254740992. ] "false:bool";
  check Ops.Cmplt Dtype.bool
    [ Const.float Dtype.float64 9007199254740992.;
      Const.int64 Dtype.weakint 9007199254740993L ] "true:bool";
  List.iter (fun (op, expected) ->
    check op Dtype.int64 [ Const.int Dtype.int64 (-7); Const.int Dtype.int64 3 ] expected)
    [ Ops.Cdiv, "-2:i64"; Ops.Cmod, "-1:i64";
      Ops.Floordiv, "-3:i64"; Ops.Floormod, "2:i64" ]

let exec_alu_weak_intermediates () =
  let dtype = Dtype.weakint in
  let value = Const.int64 dtype Int64.max_int in
  let doubled =
    match Uop.exec_alu Ops.Mul dtype [ value; Const.int dtype 2 ] with
    | Some c -> c
    | None -> fail "weak integer multiplication did not fold"
  in
  equal string "18446744073709551614:weakint" (Const.to_string doubled);
  equal (option string) (Some "9223372036854775807:weakint")
    (Option.map Const.to_string (Uop.exec_alu Ops.Sub dtype [ doubled; value ]))

let exec_alu_trunc_keeps_nonfinite () =
  let folded x =
    match Uop.exec_alu Ops.Trunc Dtype.float32 [ Const.float Dtype.float32 x ]
    with
    | Some r ->
        (match Const.view r with
         | Const.Float v -> v
         | _ -> fail "expected TRUNC to fold to a float constant")
    | None -> fail "expected TRUNC to fold"
  in
  is_true ~msg:"TRUNC rounds toward zero" (folded (-4.5) = -4.0);
  is_true ~msg:"TRUNC of +inf is +inf" (folded Float.infinity = Float.infinity);
  is_true ~msg:"TRUNC of -inf is -inf"
    (folded Float.neg_infinity = Float.neg_infinity);
  is_true ~msg:"TRUNC of nan is nan" (Float.is_nan (folded Float.nan))

let alu_unary_promotes_transcendentals () =
  let weak = Uop.const (Const.int Dtype.weakint 4) in
  is_true ~msg:"SIN of weakint promotes to weakfloat"
    (Dtype.equal (Uop.dtype (Uop.alu_unary ~op:Ops.Sin ~src:weak))
       Dtype.weakfloat);
  let typed = Uop.const (Const.int Dtype.int32 4) in
  is_true ~msg:"SQRT of an int promotes to a float"
    (Dtype.is_float (Uop.dtype (Uop.alu_unary ~op:Ops.Sqrt ~src:typed)));
  let f = Uop.const (Const.float Dtype.float32 1.0) in
  is_true ~msg:"LOG2 of a float keeps its dtype"
    (Dtype.equal (Uop.dtype (Uop.alu_unary ~op:Ops.Log2 ~src:f)) Dtype.float32);
  is_true ~msg:"NEG of an int keeps its dtype"
    (Dtype.equal (Uop.dtype (Uop.alu_unary ~op:Ops.Neg ~src:typed)) Dtype.int32);
  let raised =
    try
      ignore (Uop.alu_unary ~op:Ops.Add ~src:typed);
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"alu_unary rejects a non-unary op" raised

let division_promotes_integer_operands () =
  let divide a b = Uop.alu_binary ~op:Ops.Fdiv ~lhs:a ~rhs:b in
  let weak = divide (Uop.const_int 3) (Uop.const_int 2) in
  is_true (Dtype.equal (Uop.dtype weak) Dtype.weakfloat);
  let typed = divide (Uop.const (Const.int Dtype.int32 3))
      (Uop.const (Const.int Dtype.int32 2)) in
  is_true (Dtype.equal (Uop.dtype typed) Dtype.float32);
  (match Uop.as_const (Uop.simplify typed) with
   | Some c -> equal (option float_exact) (Some 1.5)
       (match Const.view c with Const.Float f -> Some f | _ -> None)
   | _ -> fail "integer division did not fold to a floating value")

let stack_promotes_all_operands () =
  let half = Uop.const (Const.float Dtype.float16 1.) in
  let single = Uop.const (Const.float Dtype.float32 2.) in
  List.iter (fun srcs ->
      let stacked = Uop.stack srcs in
      is_true (Dtype.equal (Uop.dtype stacked) Dtype.float32);
      equal (list int) [ 2 ] (Uop.max_shape stacked))
    [ [ half; single ]; [ single; half ] ];
  is_true (Dtype.equal (Uop.dtype (Uop.stack [])) Dtype.void)

let runtime_realization_state_parity () =
  let shape = Uop.stack [ Uop.const_int 4 ] in
  let global =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape
      ~addrspace:Dtype.Global ()
  in
  let local =
    Uop.buffer ~slot:1 ~dtype:Dtype.int32 ~shape
      ~addrspace:Dtype.Local ()
  in
  let reg =
    Uop.buffer ~slot:2 ~dtype:Dtype.int32 ~shape
      ~addrspace:Dtype.Reg ()
  in
  let param =
    Uop.param ~slot:3 ~dtype:Dtype.int32 ~shape ()
  in
  let view =
    Uop.reshape ~src:global
      ~shape:(Uop.stack [ Uop.const_int 2; Uop.const_int 2 ])
  in
  let slice =
    storage_view ~src:global ~offset:(Uop.const_int 1) ~size:2
      ~dtype:Dtype.int32
  in
  (match Uop.runtime_realization_state global with
   | Runtime_dependent [ b ] ->
       is_true ~msg:"global BUFFER depends on its runtime buffer" (b == global)
   | _ -> fail "global BUFFER should be runtime-dependent");
  (match Uop.runtime_realization_state view with
   | Runtime_dependent [ b ] ->
       is_true ~msg:"movement view asks its base BUFFER" (b == global)
   | _ -> fail "movement view should be runtime-dependent");
  is_true ~msg:"LOCAL scratch buffer is never realized"
    (Uop.runtime_realization_state local = Never_realized);
  is_true ~msg:"REG scratch buffer is never realized"
    (Uop.runtime_realization_state reg = Never_realized);
  is_true ~msg:"PARAM identity is not a runtime realized buffer"
    (Uop.runtime_realization_state param = Never_realized);
  is_true ~msg:"storage view is not a storage identity"
    (Uop.runtime_realization_state slice = Never_realized);
  let second_global =
    Uop.buffer ~slot:4 ~dtype:Dtype.int32 ~shape ()
  in
  let shard_stack = Uop.mstack [ global; second_global ] in
  (match Uop.runtime_realization_state shard_stack with
   | Runtime_dependent buffers ->
       equal int ~msg:"MSTACK depends on all global buffer sources"
         2 (List.length buffers)
   | _ -> fail "MSTACK over global buffers should be runtime-dependent");
  let scratch_stack = Uop.mstack [ global; local ] in
  is_true ~msg:"MSTACK with scratch source is never realized"
    (Uop.runtime_realization_state scratch_stack = Never_realized)

let semantic_tag_and_side_metadata () =
  let base = Uop.const_int 1 in
  let tagged = Uop.with_tag "lane0" base in
  let metadata : Uop.metadata =
    { name = "trace"; backward = false }
  in
  let with_md = Uop.with_metadata [ metadata ] base in
  is_true ~msg:"side metadata returns same node" (with_md == base);
  is_true ~msg:"metadata is stored" (Uop.metadata base = [ metadata ]);
  is_true ~msg:"side metadata excluded from semantic key"
    (Uop.semantic_key base = Uop.semantic_key with_md);
  is_true ~msg:"semantic tag changes identity" (not (Uop.equal base tagged));
  is_true ~msg:"semantic key excludes node tag"
    (Uop.semantic_key base = Uop.semantic_key tagged)

let info_function_names_follow_tinygrad () =
  let kernel_info name : Uop.kernel_info =
    {
      name;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  let program name =
    let sink = Uop.sink ~kernel_info:(kernel_info name) [] in
    Uop.program ~sink ~info:(Uop.program_info_from_sink sink) ()
  in
  let cases =
    [
      "", "";
      "9abc", "9abc";
      "kernel name", "kernel20name";
      "a-b.c", "a2Db2Ec";
      "x/y", "x2Fy";
      "\xC3\xA9", "E9";
      "\027[31mred\027[0m", "red";
    ]
  in
  List.iter
    (fun (name, expected) ->
      equal string ~msg:("KernelInfo.function_name " ^ name)
        expected (Uop.kernel_function_name (kernel_info name));
      equal string ~msg:("PROGRAM function name " ^ name)
        expected (Uop.program_function_name (program name)))
    cases

let cache_info_semantic_key_parity () =
  let kernel_info ?(beam = 0) name : Uop.kernel_info =
    {
      name;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam;
    }
  in
  let call_info ?aux () : Uop.call_info =
    {
      grad_fxn = None;
      name = Some (Uop.Label "fn");
      precompile = false;
      precompile_backward = false;
      dtype = Dtype.void;
      aux;
    }
  in
  let sink = Uop.sink ~kernel_info:(kernel_info "kernel") [] in
  let beamed = Uop.sink ~kernel_info:(kernel_info ~beam:3 "kernel") [] in
  is_true ~msg:"KernelInfo.beam participates in semantic_key"
    (Uop.semantic_key sink <> Uop.semantic_key beamed);
  let named_program name =
    let sink = Uop.sink ~kernel_info:(kernel_info name) [] in
    Uop.program ~sink ~info:(Uop.program_info_from_sink sink) ()
  in
  let raw_space = named_program "a b" and raw_hex = named_program "a20b" in
  equal string ~msg:"different raw names can share function_name"
    (Uop.program_function_name raw_space) (Uop.program_function_name raw_hex);
  is_true ~msg:"PROGRAM semantic_key keeps its kernel's raw name"
    (Uop.semantic_key raw_space <> Uop.semantic_key raw_hex);
  let call_without_aux = Uop.call ~body:sink ~args:[] ~info:(call_info ()) in
  let call_with_aux =
    Uop.call ~body:sink ~args:[] ~info:(call_info ~aux:{fallback = []; devices = ["NV"]; host = "CPU"; table = -1; inputs = []; outputs = []; timings = []; independent_accesses = []; host_deps = []; accesses = [[]]} ())
  in
  is_true ~msg:"CallInfo.aux keeps constructor identity distinct"
    (not (Uop.equal call_without_aux call_with_aux));
  is_true ~msg:"CallInfo.aux is excluded from semantic_key"
    (Uop.semantic_key call_without_aux = Uop.semantic_key call_with_aux);
  let info = Option.get (Uop.as_call_info call_with_aux) in
  let aux = { (Option.get info.aux) with fallback = [call_without_aux] } in
  let call_with_fallback = Uop.call ~body:sink ~args:[] ~info:{info with aux = Some aux} in
  let roundtrip = Uop.import (Uop.export call_with_fallback) in
  let restored = Option.get (Option.get (Uop.as_call_info roundtrip)).aux in
  is_true ~msg:"fallback graph edges remain serializable"
    (List.equal Uop.equal [call_without_aux] restored.fallback);
  is_true ~msg:"nonempty auxiliary fallback stays outside semantic_key"
    (Uop.semantic_key call_without_aux = Uop.semantic_key call_with_fallback);
  let metadata : Uop.metadata =
    { name = "trace"; backward = false }
  in
  let side_metadata = Uop.with_metadata [ metadata ] call_without_aux in
  is_true ~msg:"side metadata remains outside semantic_key"
    (Uop.semantic_key call_without_aux = Uop.semantic_key side_metadata)

let remove_all_tags_parity () =
  let metadata : Uop.metadata =
    { name = "trace"; backward = false }
  in
  let leaf =
    Uop.with_metadata [ metadata ] (Uop.with_tag "leaf" (Uop.const_int 1))
  in
  let body =
    Uop.with_tag "body"
      (Uop.sink [ Uop.with_tag "inside-call" (Uop.const_int 2) ])
  in
  let info : Uop.call_info =
    {
      grad_fxn = None;
      name = None;
      precompile = false;
      precompile_backward = false;
      dtype = Dtype.void;
      aux = None;
    }
  in
  let root =
    Uop.with_tag "root"
      (Uop.call ~body ~args:[ Uop.with_tag "arg" leaf ] ~info)
  in
  let stripped = Uop.remove_all_tags root in
  is_true ~msg:"remove_all_tags preserves semantic cache key"
    (Uop.semantic_key root = Uop.semantic_key stripped);
  List.iter
    (fun u -> is_true ~msg:"reachable node tag cleared" (Uop.node_tag u = None))
    (Uop.toposort ~enter_calls:true stripped);
  let tagged_nodes =
    List.filter_map
      (fun u ->
        match Uop.metadata u with
        | [] -> None
        | md -> Some md)
      (Uop.toposort ~enter_calls:true stripped)
  in
  equal (list string) ~msg:"side metadata survives tag stripping"
    [ metadata.name ]
    (List.concat_map (List.map (fun (m : Uop.metadata) -> m.name)) tagged_nodes);
  let untagged =
    Uop.alu_binary ~op:Ops.Add ~lhs:(Uop.const_int 4) ~rhs:(Uop.const_int 3)
  in
  is_true ~msg:"remove_all_tags leaves untagged graphs unchanged"
    (Uop.remove_all_tags untagged == untagged)

let program_constructor_prefix_layouts () =
  let sink = Uop.sink [] in
  let linear = Uop.linear [] in
  let source = Uop.source "src" in
  let binary = Uop.binary "bin" in
  let info : Uop.program_info =
    {
      target = Target.of_string "";
      global_size = [ Launch_int 1; Launch_int 1; Launch_int 1 ];
      local_size = [ Uop.Launch_int 1; Uop.Launch_int 1; Uop.Launch_int 1 ];
      vars = [];
      globals = [];
      outs = [];
      ins = [];
    }
  in
  let src_ops u = Array.to_list (Uop.src u) |> List.map Uop.op in
  equal (list op_testable) ~msg:"Program(SINK)"
    [ Ops.Sink ] (src_ops (Uop.program ~sink ~info ()));
  equal (list op_testable) ~msg:"Program(SINK, LINEAR)"
    [ Ops.Sink; Ops.Linear ] (src_ops (Uop.program ~sink ~linear ~info ()));
  equal (list op_testable) ~msg:"Program(SINK, LINEAR, SOURCE)"
    [ Ops.Sink; Ops.Linear; Ops.Source ]
    (src_ops (Uop.program ~sink ~linear ~source ~info ()));
  equal (list op_testable) ~msg:"Program(SINK, LINEAR, SOURCE, BINARY)"
    [ Ops.Sink; Ops.Linear; Ops.Source; Ops.Binary ]
    (src_ops (Uop.program ~sink ~linear ~source ~binary ~info ()));
  let rejects f =
    try
      ignore (f ());
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"Program rejects skipped LINEAR"
    (rejects (fun () -> Uop.program ~sink ~source ~info ()));
  is_true ~msg:"Program rejects skipped SOURCE"
    (rejects (fun () -> Uop.program ~sink ~linear ~binary ~info ()))

let program_info_from_sink_parity () =
  let core_id =
    Uop.param ~slot:0 ~dtype:Dtype.weakint ~name:"core_id"
      ~vmin_vmax:(Bound.int (0), Bound.int (3)) ~addrspace:Dtype.Alu ()
  in
  let n =
    Uop.param ~slot:1 ~dtype:Dtype.weakint ~name:"n"
      ~vmin_vmax:(Bound.int (2), Bound.int (8)) ~addrspace:Dtype.Alu ()
  in
  let input = Uop.param ~slot:2 ~dtype:Dtype.float32 () in
  let output = Uop.param ~slot:3 ~dtype:Dtype.float32 () in
  let input_idx = Uop.index ~ptr:input ~idxs:[n] () in
  let output_idx = Uop.index ~ptr:output ~idxs:[n] () in
  let loaded = Uop.load ~src:input_idx () in
  let stored = Uop.store ~dst:output_idx ~value:loaded () in
  let group_dim =
    Uop.special ~name:"gidx2" ~size:(Uop.O.(n + Uop.const_int 1)) ()
  in
  let local_dim =
    Uop.special ~name:"lidx1" ~size:(Uop.const_int 8) ()
  in
  let kernel_info : Uop.kernel_info =
    {
      name = "kernel name";
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  let sink = Uop.sink ~kernel_info [ stored; group_dim; local_dim; core_id ] in
  let info = Uop.program_info_from_sink sink in
  equal string ~msg:"PROGRAM name comes from KernelInfo"
    "kernel20name" (Uop.program_function_name (Uop.program ~sink ~info ()));
  equal (list int) ~msg:"ProgramInfo globals"
    [ 2; 3 ] info.globals;
  equal (list int) ~msg:"ProgramInfo outs"
    [ 3 ] info.outs;
  equal (list int) ~msg:"ProgramInfo ins"
    [ 2 ] info.ins;
  equal int ~msg:"ProgramInfo vars count" 2 (List.length info.vars);
  is_true ~msg:"ProgramInfo vars sorted by slot"
    (List.hd info.vars == core_id && List.nth info.vars 1 == n);
  equal (list int64) ~msg:"ProgramInfo values include every scalar"
    [ 2L; 6L ] (Uop.program_vals info ~var_vals:[ "core_id", 2L; "n", 6L ]);
  raises_match
    (function
      | Invalid_argument msg -> contains msg "\"n\""
      | _ -> false)
    (fun () -> Uop.program_vals info ~var_vals:[ "core_id", 2L ]);
  raises_match
    (function
      | Invalid_argument msg -> contains msg "\"n\""
      | _ -> false)
    (fun () -> Uop.program_launch_dims info ~var_vals:[]);
  let global_size, local_size =
    Uop.program_launch_dims info ~var_vals:[ "n", 6L ]
  in
  equal (list launch_value_testable) ~msg:"ProgramInfo launch dims"
    [ Launch_value_int 1; Launch_value_int 1; Launch_value_int 7 ]
    global_size;
  equal (list launch_value_testable) ~msg:"ProgramInfo local dims"
    [ Launch_value_int 1; Launch_value_int 8; Launch_value_int 1 ] local_size

let program_launch_dims_floor_divmod () =
  let n =
    Uop.param ~slot:0 ~dtype:Dtype.weakint ~name:"n"
      ~vmin_vmax:(Bound.int (-10), Bound.int (10)) ~addrspace:Dtype.Alu ()
  in
  let three = Uop.const_int 3 in
  let groups = Uop.O.(n // three) in
  let groups_y = Uop.O.(n mod three) in
  let group_dim = Uop.special ~name:"gidx0" ~size:groups () in
  let group_dim_y = Uop.special ~name:"gidx1" ~size:groups_y () in
  let sink = Uop.sink [ group_dim; group_dim_y ] in
  let info = Uop.program_info_from_sink sink in
  let global_size, local_size =
    Uop.program_launch_dims info ~var_vals:[ "n", -7L ]
  in
  equal (list launch_value_testable)
    ~msg:"ProgramInfo floor launch global dims"
    [ Launch_value_int (-3); Launch_value_int 2; Launch_value_int 1 ]
    global_size;
  equal (list launch_value_testable) ~msg:"ProgramInfo floor launch local dims"
    [ Launch_value_int 1; Launch_value_int 1; Launch_value_int 1 ] local_size

let sym_infer_host_scalars () =
  let n = Uop.variable ~name:"host_n" ~min_val:(-1000) ~max_val:max_int () in
  let half = Uop.alu_binary ~op:Ops.Fdiv
      ~lhs:(Uop.cast ~src:n ~dtype:Dtype.float32) ~rhs:(Uop.const_float 2.) in
  let truncated = Uop.cast ~src:half ~dtype:Dtype.weakint in
  equal int 7 (Uop.sym_infer Uop.O.(truncated + Uop.const_int 10) ["host_n", -7L]);
  equal int 0 (Uop.sym_infer Uop.O.(n * Uop.const_int 0) []);
  let bound = Uop.bind ~var:n ~value:(Uop.const_int 7) in
  equal int 9 (Uop.sym_infer Uop.O.(bound + Uop.const_int 1) ["host_n", 8L]);
  raises (Invalid_argument "sym_infer: result does not fit a host integer")
    (fun () -> Uop.sym_infer Uop.O.(n * n) ["host_n", 0x1_0000_0000L]);
  raises (Invalid_argument "sym_infer: missing variable \"host_n\"")
    (fun () -> Uop.sym_infer n [])

let debug_prints_toposort_like_tinygrad () =
  let a = Uop.const_int 1 in
  let b = Uop.const_int 2 in
  let add = Uop.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let expected =
    "   0 Ops.CONST           :            dtypes.weakint                           []                               1\n\
     \   1 Ops.CONST           :            dtypes.weakint                           []                               2\n\
     \   2 Ops.ADD             :            dtypes.weakint                           ['1', '2']                       None\n"
  in
  equal text expected (Render.uops_to_string add)

let debug_prints_ranges_and_supplied_list_sources () =
  let r =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Weak ()
  in
  let expr = Uop.O.(r + Uop.const_int 1) in
  let expected_toposort =
    "   0 Ops.CONST           :            dtypes.weakint                           []                               4\n\
     \   1 Ops.RANGE           : 0          dtypes.weakint                           ['4']                            (0, AxisType.WEAK)\n\
     \   2 Ops.CONST           :            dtypes.weakint                           []                               1\n\
     \   3 Ops.ADD             : 0          dtypes.weakint                           [1, '1']                         None\n"
  in
  equal text expected_toposort (Render.uops_to_string expr);
  let expected_list =
    "   0 Ops.ADD             : 0          dtypes.weakint                           [1, '--']                        None\n\
     \   1 Ops.RANGE           : 0          dtypes.weakint                           ['--']                           (0, AxisType.WEAK)\n"
  in
  equal text expected_list (Render.uops_list_to_string [ expr; r ])

let debug_prints_tinygrad_dtype_reprs () =
  let vec = Uop.stack [ Uop.const_int 1; Uop.const_int 2 ] in
  let buffer = Uop.buffer ~slot:0 ~dtype:Dtype.float32 () in
  let long = Uop.buffer ~slot:1 ~dtype:Dtype.int64 () in
  is_true ~msg:"weakint dtype repr"
    (contains (Render.uops_to_string vec) "dtypes.weakint");
  is_true ~msg:"scalar float dtype repr"
    (contains (Render.uops_to_string buffer) "dtypes.float");
  is_true ~msg:"scalar long dtype repr"
    (contains (Render.uops_to_string long) "dtypes.long")

let debug_prints_float_and_special_args_like_tinygrad () =
  let float_one = Uop.const_float 1.0 in
  let special =
    Uop.special ~name:"lidx0" ~size:(Uop.const_int 8) ()
  in
  let out = Render.uops_list_to_string [ float_one; special ] in
  (match Uop.arg special with
   | Uop.Arg.String "lidx0" -> ()
   | _ -> is_true ~msg:"SPECIAL stores raw string arg" false);
  is_true ~msg:"float repr keeps .0" (contains out "1.0\n");
  is_true ~msg:"SPECIAL arg is raw" (contains out " lidx0\n");
  is_true ~msg:"SPECIAL arg is not quoted" (not (contains out "'lidx0'"))

let debug_prints_direct_string_args_like_tinygrad () =
  let src = Uop.source "abc" in
  let copied = Uop.copy ~src:(Uop.const (Const.int Dtype.int32 1)) ~device:(Uop.Single "CPU") () in
  let out = Render.uops_list_to_string [ src; copied ] in
  is_true ~msg:"SOURCE string arg is raw" (contains out " abc\n");
  is_true ~msg:"COPY string device arg is raw" (contains out " CPU\n");
  is_true ~msg:"direct string args are not quoted" (not (contains out "'abc'"));
  is_true ~msg:"direct device args are not quoted" (not (contains out "'CPU'"))

let debug_prints_ranges_in_tinygrad_arg_order () =
  let size = Uop.const_int 4 in
  let r_sub1 =
    Uop.range ~size ~axis:0 ~sub:[ 1 ] ~kind:Axis_type.Global ()
  in
  let r_sub0 =
    Uop.range ~size ~axis:0 ~sub:[ 0 ] ~kind:Axis_type.Upcast ()
  in
  let expr = Uop.O.(r_sub1 + r_sub0) in
  is_true ~msg:"ranges sort by raw arg order"
    (contains (Render.uops_to_string expr) "0_0,0_1")

let debug_prints_rich_args_dataclass_style () =
  let param =
    Uop.variable ~name:"n" ~min_val:1 ~max_val:9 ()
  in
  let buffer =
    Uop.buffer ~slot:2 ~dtype:Dtype.int32 ~name:"buf"
      ~addrspace:Dtype.Local ()
  in
  let staged =
    Uop.stage ~src:(Uop.const_int 1) ~ranges:[]
      ~opts:
        { device = Some (Uop.Index 0); addrspace = Dtype.Local; removable = false }
  in
  let kernel_info : Uop.kernel_info =
    {
      name = "kern";
      applied_opts = [ Uop.Opt.Split { kind = Axis_type.Upcast; top = false; axis = 0; amount = 4 } ];
      opts_to_apply = Some [ Uop.Opt.Padto { axis = 0; amount = 4 } ];
      estimates = Some { ops = Uop.Int 1; lds = Uop.Int 2; mem = Uop.Int 3 };
      beam = 0;
    }
  in
  let sink = Uop.sink ~kernel_info [] in
  let program_info : Uop.program_info =
    {
      target = Target.of_string "";
      global_size = [ Launch_int 1; Launch_float 2.0 ];
      local_size = [ Uop.Launch_int 1; Uop.Launch_int 1; Uop.Launch_int 1 ];
      vars = [ param ];
      globals = [ 2 ];
      outs = [];
      ins = [ 2 ];
    }
  in
  let program = Uop.program ~sink ~info:program_info () in
  let info =
    {
      Uop.grad_fxn = None;
      name = Some (Uop.Label "fn");
      precompile = true;
      precompile_backward = false;
      dtype = Dtype.void;
      aux = None;
    }
  in
  let call = Uop.call ~body:sink ~args:[] ~info in
  let out =
    Render.uops_list_to_string [ param; buffer; staged; sink; program; call ]
  in
  is_true ~msg:"ParamArg repr"
    (contains out
       "ParamArg(-1, dtypes.weakint, vmin_vmax=(1, 9), multiple_of=1, name='n', addrspace=AddrSpace.ALU)");
  is_true ~msg:"Buffer ParamArg repr"
    (contains out
       "ParamArg(2, dtypes.int, name='buf', addrspace=AddrSpace.LOCAL)");
  is_true ~msg:"Stage repr"
    (contains out
       "BufferizeOpts(device=0, addrspace=AddrSpace.LOCAL, removable=False)");
  is_true ~msg:"KernelInfo repr"
    (contains out
       "KernelInfo(name='kern', applied_opts=");
  is_true ~msg:"Opt repr"
    (contains out "Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))");
  is_true ~msg:"Estimates repr"
    (contains out "Estimates(ops=1, lds=2, mem=3)");
  is_true ~msg:"ProgramInfo repr"
    (contains out
       "ProgramInfo(global_size=(1, 2.0), local_size=(1, 1, 1)");
  is_true ~msg:"ProgramInfo vars use UOp repr"
    (contains out "vars=(UOp(Ops.BUFFER, dtypes.weakint, arg=ParamArg");
  is_true ~msg:"CallInfo repr"
    (contains out "CallInfo(None, 'fn', True, False)")

let debug_prints_reduce_arg_tuple () =
  let body =
    Uop.buffer ~slot:0 ~dtype:Dtype.float32
      ~shape:(Uop.stack [ Uop.const_int 4 ]) ()
  in
  let tensor_reduce = Uop.reduce_axis ~src:body ~op:Ops.Add ~axes:[ 0 ] in
  let range =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Reduce ()
  in
  let lowered =
    Uop.reduce ~src:body ~ranges:[ range ] ~op:Ops.Add
  in
  let out = Render.uops_list_to_string [ tensor_reduce; lowered ] in
  is_true ~msg:"tensor reduce arg repr"
    (contains out "(Ops.ADD, 1)");
  is_true ~msg:"lowered reduce arg repr"
    (contains out "(Ops.ADD, 0)")

let debug_print_ignores_side_metadata () =
  let base = Uop.const_int 9 in
  let before = Render.uops_to_string base in
  let metadata : Uop.metadata =
    { name = "trace"; backward = true }
  in
  ignore (Uop.with_metadata [ metadata ] base);
  equal text before (Render.uops_to_string base)

let debug_listing_omits_tags () =
  let leaf = Uop.with_tag "golden-leaf" (Uop.const_int 303) in
  let root =
    Uop.with_tag "golden-root"
      (Uop.alu_binary ~op:Ops.Add ~lhs:leaf ~rhs:(Uop.const_int 404))
  in
  let listing = Render.uops_to_string root in
  is_true ~msg:"debug listing omits node tags"
    (not (contains listing "golden-root")
     && not (contains listing "golden-leaf"))

(* Upat *)

let upat_matches_add () =
  let u = Uop.alu_binary ~op:Ops.Add ~lhs:(Uop.const_int 1) ~rhs:(Uop.const_int 2) in
  let p = Upat.op Ops.Add in
  is_true ~msg:"matches" (Upat.match_ p u <> [])

let upat_captures_operands () =
  let u = Uop.alu_binary ~op:Ops.Add ~lhs:(Uop.const_int 7) ~rhs:(Uop.const_int 9) in
  let p = Upat.op ~src:[ Upat.var "x"; Upat.var "y" ] Ops.Add in
  match Upat.match_ p u with
  | bs :: _ ->
      let x = Upat.(bs $ "x") and y = Upat.(bs $ "y") in
      let get_int u = match Uop.arg u with
        | Uop.Arg.Value c -> (match Const.view c with
            | Int n -> Some (Z.to_int n) | _ -> None)
        | _ -> None
      in
      is_true ~msg:"x = 7" (get_int x = Some 7);
      is_true ~msg:"y = 9" (get_int y = Some 9)
  | [] -> is_true ~msg:"match succeeded" false

let pattern_matcher_rewrites () =
  let open Upat in
  let rules = Pattern_matcher.make [
    O.(var "x" + zero) => (fun bs -> Some (bs $ "x"));
    O.(var "x" * one)  => (fun bs -> Some (bs $ "x"));
  ] in
  let x = Uop.const_int 42 in
  let e1 = Uop.alu_binary ~op:Ops.Add ~lhs:x ~rhs:(Uop.const_int 0) in
  let e2 = Uop.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:(Uop.const_int 1) in
  let fired_with_x e =
    match Pattern_matcher.rewrite rules e with
    | Some r -> r == x
    | None -> false
  in
  is_true ~msg:"x + 0 -> x" (fired_with_x e1);
  is_true ~msg:"x * 1 -> x" (fired_with_x e2)

let upat_operator_surface_matches_tinygrad () =
  let open Upat in
  let x = Uop.variable ~name:"x" ~min_val:1 ~max_val:64 () in
  let y = Uop.variable ~name:"y" ~min_val:1 ~max_val:64 () in
  let matches pat u = match_ pat u <> [] in
  is_true ~msg:"/ pattern matches true division"
    (matches O.(var "x" / var "y")
       (Uop.alu_binary ~op:Ops.Fdiv ~lhs:x ~rhs:y));
  is_true ~msg:"// pattern matches floor division"
    (matches O.(var "x" // var "y")
       (Uop.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:y));
  is_true ~msg:"/ pattern does not match truncating division"
    (not
       (matches O.(var "x" / var "y")
          (Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:y)));
  is_true ~msg:"mod pattern matches floor modulo"
    (matches O.(var "x" mod var "y")
       (Uop.alu_binary ~op:Ops.Floormod ~lhs:x ~rhs:y));
  is_true ~msg:"explicit cdiv pattern matches truncating division"
    (matches O.(cdiv (var "x") (var "y"))
       (Uop.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:y));
  is_true ~msg:"explicit cmod pattern matches truncating modulo"
    (matches O.(cmod (var "x") (var "y"))
       (Uop.alu_binary ~op:Ops.Cmod ~lhs:x ~rhs:y))

let pattern_matcher_context_rewrites () =
  let open Upat in
  let rules =
    Pattern_matcher.make_with_ctx [
      with_ctx (op ~name:"u" Ops.Const) (fun replacement bs ->
        let matched = bs $ "u" in
        if Uop.op matched = Ops.Const then Some replacement else None);
    ]
  in
  let replacement = Uop.const_int 7 in
  match Pattern_matcher.rewrite_with_ctx rules ~ctx:replacement (Uop.const_int 1) with
  | Some r -> is_true ~msg:"context reaches callback" (r == replacement)
  | None -> is_true ~msg:"context rule fired" false

let upat_matches_node_tags () =
  let tagged = Uop.with_tag "lane0" (Uop.const_int 1) in
  is_true ~msg:"matching tag succeeds"
    (Upat.match_ (Upat.tag "lane0" (Upat.cvar ())) tagged <> []);
  is_true ~msg:"different tag does not match"
    (Upat.match_ (Upat.tag "lane1" (Upat.cvar ())) tagged = [])

let upat_numeric_literals () =
  let matches pattern value = Upat.match_ pattern (Uop.const value) <> [] in
  let exact = Const.integer Dtype.weakint (Z.of_string "9007199254740992") in
  let next = Const.integer Dtype.weakint (Z.of_string "9007199254740993") in
  let rounded = Const.float Dtype.weakfloat 9007199254740992. in
  is_true ~msg:"equal integer and float match" (matches (Upat.const exact) rounded);
  is_false ~msg:"integer is not rounded to match float"
    (matches (Upat.const next) rounded);
  is_false ~msg:"float pattern does not round integer"
    (matches (Upat.const rounded) next);
  is_true ~msg:"negative zero matches integer zero"
    (matches Upat.zero (Const.float Dtype.weakfloat (-0.)));
  is_true ~msg:"negative zero matches positive zero"
    (matches (Upat.const_float 0.) (Const.float Dtype.weakfloat (-0.)));
  is_true ~msg:"integer one matches bool true" (matches Upat.one (Const.bool true));
  is_false ~msg:"fractional floats do not match integers"
    (matches Upat.one (Const.float Dtype.weakfloat 1.5));
  is_false ~msg:"infinity does not match integer"
    (matches (Upat.const exact) (Const.float Dtype.weakfloat infinity))

let pattern_matcher_rejects_opless_rules () =
  let open Upat in
  let rejected =
    try
      let rules =
        Pattern_matcher.make [
          any => (fun bs -> if mem bs "unused" then Some (Uop.const_int 0) else None);
        ]
      in
      ignore rules;
      false
    with Invalid_argument msg ->
      String.length msg > 0
  in
  is_true ~msg:"op-less root pattern is rejected by make" rejected

let context_matcher_rejects_opless_rules () =
  let open Upat in
  let rejected =
    try
      let rules =
        Pattern_matcher.make_with_ctx [
          with_ctx any (fun () bs ->
            if mem bs "unused" then Some (Uop.const_int 0) else None);
        ]
      in
      ignore rules;
      false
    with Invalid_argument msg ->
      String.length msg > 0
  in
  is_true ~msg:"op-less root pattern is rejected by make_with_ctx" rejected

let custom_early_reject_skips_callback () =
  let open Upat in
  let rules =
    Pattern_matcher.make [
      early_reject [ Ops.Mul ] (op Ops.Add) => (fun bs ->
        if mem bs "unreachable" then None
        else failwith "custom early reject did not skip callback");
    ]
  in
  let e = Uop.O.(Uop.const_int 1 + Uop.const_int 2) in
  is_true ~msg:"early reject prevents callback"
    (Pattern_matcher.rewrite rules e = None)

let upat_dtype_matches_scalar_of_vector () =
  let v = Uop.stack [ Uop.const_int 1; Uop.const_int 2 ] in
  let cmp = Uop.O.(v < v) in
  let p = Upat.var_dtype "x" (Upat.exact_dtype Dtype.Bool) in
  is_true ~msg:"bool pattern matches bool comparison dtype"
    (Upat.match_ p cmp <> [])

let upat_explicit_source_patterns () =
  let a = Uop.const_int 1
  and b = Uop.const_int 2
  and c = Uop.const_int 3 in
  let sum = Uop.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let sink = Uop.sink [ a; b; c ] in
  let vector = Uop.stack [ a; a; a ] in
  is_true ~msg:"fixed source pattern matches exact sources"
    (Upat.match_
       (Upat.op_src ~src:(Upat.fixed [ Upat.const_int 1; Upat.const_int 2 ])
          Ops.Add)
       sum
     <> []);
  is_true ~msg:"perms source pattern matches commuted operands"
    (Upat.match_
       (Upat.op_src ~src:(Upat.perms [ Upat.const_int 2; Upat.const_int 1 ])
          Ops.Add)
       sum
     <> []);
  is_true ~msg:"fixed source pattern accepts trailing sources"
    (Upat.match_
       (Upat.op_src ~allow_any_len:true
          ~src:(Upat.fixed [ Upat.const_int 1; Upat.const_int 2 ])
          Ops.Sink)
       sink
     <> []);
  is_true ~msg:"repeat source pattern matches every source"
    (Upat.match_
       (Upat.op_src ~src:(Upat.repeat (Upat.const_int 1)) Ops.Stack)
       vector
     <> []);
  is_true ~msg:"is_any pattern chooses matching branch"
    (Upat.match_ (Upat.is_any [ Upat.const_int 0; Upat.const_int 3 ]) c <> [])

let upat_matches_reduce_arg () =
  let body =
    Uop.buffer ~slot:0 ~dtype:Dtype.float32
      ~shape:(Uop.stack [ Uop.const_int 4 ]) ()
  in
  let red = Uop.reduce_axis ~src:body ~op:Ops.Add ~axes:[ 0 ] in
  is_true ~msg:"dedicated reduce op predicate matches"
    (Upat.match_
       (Upat.op ~arg:(Upat.has_reduce_op Ops.Add) Ops.Reduce)
       red
     <> []);
  is_true ~msg:"old Op predicate does not match reduce arg"
    (Upat.match_ (Upat.op ~arg:(Upat.has_op Ops.Add) Ops.Reduce) red = [])

let upat_rejects_reserved_ctx_capture () =
  let rejected =
    try
      ignore (Upat.var "ctx");
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"ctx capture name is reserved" rejected

let upat_permutation_matches_are_deduplicated () =
  let x = Uop.const_int 1 in
  let sum = Uop.alu_binary ~op:Ops.Add ~lhs:x ~rhs:x in
  let pat =
    Upat.op_src
      ~src:(Upat.perms [ Upat.var "x"; Upat.var "y" ])
      Ops.Add
  in
  is_true ~msg:"duplicate permutation bindings collapsed"
    (List.length (Upat.match_ pat sum) = 1)

let pattern_matcher_ignores_self_replacement () =
  let open Upat in
  let x = Uop.const_int 7 in
  let rules =
    Pattern_matcher.make
      [
        const_int 7 => (fun _ -> Some x);
        const_int 7 => (fun _ -> Some (Uop.const_int 8));
      ]
  in
  match Pattern_matcher.rewrite rules x with
  | Some r -> is_true ~msg:"self replacement skipped" (Uop.equal r (Uop.const_int 8))
  | None -> is_true ~msg:"second rule fired after self replacement" false

let graph_rewrite_walk_does_not_enter_replacements () =
  let rewrite u =
    match Uop.arg u with
    | Uop.Arg.Value c ->
        (match Const.view c with
         | Int n when Z.equal n Z.one ->
             Some Uop.O.(Uop.const_int 2 + Uop.const_int 3)
         | Int n when Z.equal n (Z.of_int 2) -> Some (Uop.const_int 20)
         | view ->
             ignore view;
             None)
    | arg ->
        ignore arg;
        None
  in
  let r = Uop.graph_rewrite ~walk:true rewrite (Uop.const_int 1) in
  match Uop.op r, Array.to_list (Uop.src r) with
  | Ops.Add, lhs :: rhs :: rest ->
      is_true ~msg:"replacement child left untouched" (Uop.equal lhs (Uop.const_int 2));
      is_true ~msg:"replacement rhs preserved" (Uop.equal rhs (Uop.const_int 3));
      is_true ~msg:"binary add has no extra srcs" (rest = [])
  | op, srcs ->
      ignore op;
      ignore srcs;
      is_true ~msg:"walk rewrite returns replacement Add" false

let graph_rewrite_bpm_runs_before_post_order () =
  let label u =
    match Uop.const_int_value u with
    | Some n -> Printf.sprintf "CONST:%d" n
    | None -> Ops.name (Uop.op u)
  in
  let events = ref [] in
  let record prefix u =
    events := Printf.sprintf "%s:%s" prefix (label u) :: !events
  in
  let bpm u =
    record "bpm" u;
    None
  in
  let post u =
    record "post" u;
    None
  in
  let root = Uop.O.(Uop.const_int 1 + Uop.const_int 2) in
  ignore (Uop.graph_rewrite ~bpm post root);
  is_true ~msg:"bpm runs pre-order and post runs after rewritten children"
    (List.rev !events
     = [
         "bpm:ADD";
         "bpm:CONST:1";
         "post:CONST:1";
         "bpm:CONST:2";
         "post:CONST:2";
         "post:ADD";
       ])

let graph_rewrite_walk_bpm_short_circuits_replacement () =
  let bpm u =
    match Uop.const_int_value u with
    | Some 1 -> Some Uop.O.(Uop.const_int 3 + Uop.const_int 4)
    | Some n ->
        ignore n;
        None
    | None -> None
  in
  let post u =
    match Uop.const_int_value u with
    | Some 3 -> Some (Uop.const_int 30)
    | Some n ->
        ignore n;
        None
    | None -> None
  in
  let root = Uop.O.(Uop.const_int 1 + Uop.const_int 2) in
  let r = Uop.graph_rewrite ~walk:true ~bpm post root in
  match Array.to_list (Uop.src r) with
  | replacement :: rhs :: rest -> (
      is_true ~msg:"walk pre-match replacement became lhs"
        (Ops.equal (Uop.op replacement) Ops.Add);
      is_true ~msg:"original rhs is preserved" (Uop.equal rhs (Uop.const_int 2));
      is_true ~msg:"root add has two sources" (rest = []);
      match Array.to_list (Uop.src replacement) with
      | lhs :: mid_rhs :: inner_rest ->
          is_true ~msg:"replacement lhs was not post-rewritten"
            (Uop.equal lhs (Uop.const_int 3));
          is_true ~msg:"replacement rhs was preserved"
            (Uop.equal mid_rhs (Uop.const_int 4));
          is_true ~msg:"replacement add has two sources" (inner_rest = [])
      | inner_srcs ->
          ignore inner_srcs;
          is_true ~msg:"replacement is binary Add" false)
  | srcs ->
      ignore srcs;
      is_true ~msg:"root is binary Add" false

let graph_rewrite_bottom_up_gate_skips_post_and_children () =
  let post_called = ref false in
  let child_called = ref false in
  let root = Uop.O.(Uop.const_int 1 + Uop.const_int 2) in
  let bpm u =
    if Uop.equal u root then raise Uop.Bottom_up_gate;
    child_called := true;
    None
  in
  let post u =
    post_called := true;
    Some u
  in
  let r = Uop.graph_rewrite ~bpm post root in
  is_true ~msg:"gated root is preserved" (Uop.equal r root);
  is_true ~msg:"gated node children are skipped" (not !child_called);
  is_true ~msg:"gated node post-match is skipped" (not !post_called)

let graph_rewrite_walk_bottom_up_gate_skips_post_and_children () =
  let post_called = ref false in
  let child_called = ref false in
  let root = Uop.O.(Uop.const_int 1 + Uop.const_int 2) in
  let bpm u =
    if Uop.equal u root then raise Uop.Bottom_up_gate;
    child_called := true;
    None
  in
  let post u =
    post_called := true;
    Some u
  in
  let r = Uop.graph_rewrite ~walk:true ~bpm post root in
  is_true ~msg:"walk gated root is preserved" (Uop.equal r root);
  is_true ~msg:"walk gated node children are skipped" (not !child_called);
  is_true ~msg:"walk gated node post-match is skipped" (not !post_called)

let graph_rewrite_skips_call_body_by_default () =
  let info =
    {
      Uop.grad_fxn = None;
      name = None;
      precompile = false;
      precompile_backward = false;
      dtype = Dtype.void;
      aux = None;
    }
  in
  let body = Uop.sink [ Uop.const_int 1 ] in
  let arg = Uop.const_int 2 in
  let call = Uop.call ~body ~args:[ arg ] ~info in
  let rewrite u =
    match Uop.arg u with
    | Uop.Arg.Value c ->
        (match Const.view c with
         | Int n when Z.equal n Z.one -> Some (Uop.const_int 10)
         | Int n when Z.equal n (Z.of_int 2) -> Some (Uop.const_int 20)
         | view ->
             ignore view;
             None)
    | arg ->
        ignore arg;
        None
  in
  let r = Uop.graph_rewrite rewrite call in
  match Array.to_list (Uop.src r) with
  | body' :: arg' :: rest ->
      is_true ~msg:"call body is not entered" (body' == body);
      is_true ~msg:"call arg is still rewritten" (Uop.equal arg' (Uop.const_int 20));
      is_true ~msg:"call has no extra srcs" (rest = [])
  | srcs ->
      ignore srcs;
      is_true ~msg:"call keeps body and one arg" false

(* Skipping the body is a property of the body, not of the edge that reaches
   it: once a call pins its body, a second path to that same body is left
   alone too. Otherwise a body shared with the surrounding graph would be
   rewritten through the other path and the call would no longer name it. *)
let graph_rewrite_pins_call_body_on_every_path () =
  let info =
    {
      Uop.grad_fxn = None;
      name = None;
      precompile = false;
      precompile_backward = false;
      dtype = Dtype.void;
      aux = None;
    }
  in
  let body = Uop.sink [ Uop.const_int 1 ] in
  let call = Uop.call ~body ~args:[ Uop.const_int 2 ] ~info in
  let rewrite u =
    match Uop.const_int_value u with
    | Some 1 -> Some (Uop.const_int 10)
    | Some 2 -> Some (Uop.const_int 20)
    | _ -> None
  in
  let r = Uop.graph_rewrite rewrite (Uop.sink [ call; body ]) in
  match Array.to_list (Uop.src r) with
  | [ call'; body' ] ->
      is_true ~msg:"call body is not entered" ((Uop.src call').(0) == body);
      is_true ~msg:"call arg is still rewritten"
        (Uop.equal (Uop.src call').(1) (Uop.const_int 20));
      is_true ~msg:"the other path to the body is not entered either"
        (body' == body)
  | srcs ->
      ignore srcs;
      is_true ~msg:"sink keeps the call and the body" false

let graph_rewrite_enters_native_callee () =
  let dependency = Uop.const_int 1 in
  let pointer = Uop.after ~src:(Uop.const (Const.int Dtype.uint64 8)) ~deps:[dependency] in
  let body = Uop.custom_function ~name:"native" ~srcs:[pointer] in
  let call = Uop.call ~body ~args:[]
      ~info:{grad_fxn = None; name = None; precompile = false;
        precompile_backward = false; dtype = Dtype.void; aux = None} in
  let rewrite u = if u == dependency then Some (Uop.const_int 2) else None in
  let result = Uop.graph_rewrite rewrite (Uop.sink [call; dependency]) in
  let call = (Uop.src result).(0) in
  let pointer = (Uop.src (Uop.src call).(0)).(0) in
  is_true ~msg:"native callee dependencies share the caller rewrite"
    ((Uop.src pointer).(1) == (Uop.src result).(1))

let swap_one_and_two u =
  match Uop.const_int_value u with
  | Some 1 -> Some (Uop.const_int 2)
  | Some 2 -> Some (Uop.const_int 1)
  | _ -> None

let graph_rewrite_detects_bottom_up_cycles () =
  let rejected =
    try
      ignore
        (Uop.graph_rewrite ~bottom_up:true swap_one_and_two (Uop.const_int 1));
      false
    with Invalid_argument msg ->
      contains msg "cycle"
  in
  is_true ~msg:"bottom-up rewrite cycle is rejected" rejected

(* A node's replacement is itself rewritten, so a rule that maps a pair of
   nodes onto each other never settles. It is reported where it closes, not
   left to run away. *)
let graph_rewrite_detects_top_down_cycles () =
  let rejected =
    try
      ignore (Uop.graph_rewrite swap_one_and_two (Uop.const_int 1));
      false
    with Invalid_argument msg ->
      contains msg "cycle"
  in
  is_true ~msg:"top-down rewrite cycle is rejected" rejected

let after_closes_ranges_from_dependencies () =
  let r =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Weak ()
  in
  let ended = Uop.end_ ~value:(Uop.const_int 0) ~ranges:[ r ] in
  let sequenced = Uop.after ~src:r ~deps:[ ended ] in
  is_true ~msg:"dependency-ended range is not live"
    (not (List.exists (Uop.equal r) (Uop.ranges sequenced)))

let shared_ending_dependencies_have_bounded_allocations () =
  let leaf = Uop.barrier ~srcs:[Uop.const_int 91283] () in
  let rec nest depth node =
    if depth = 0 then node
    else nest (depth - 1) (Uop.barrier ~srcs:[node; node] ()) in
  let root = nest 16 leaf in
  let minor_before, promoted_before, major_before = Gc.counters () in
  let ranges = Uop.ranges root in
  let minor_after, promoted_after, major_after = Gc.counters () in
  let words = minor_after -. minor_before +. major_after -. major_before
      -. (promoted_after -. promoted_before) in
  equal int 0 (List.length ranges);
  if words > 50_000. then
    failf "17 shared barriers allocated %.0f words while resolving empty ranges" words

let nested_ending_dependencies_preserve_live_range_order () =
  let range axis = Uop.range ~size:(Uop.const_int 4) ~axis ~kind:Axis_type.Weak () in
  let first = range 91284 and closed = range 91285 and last = range 91286 in
  let end_closed = Uop.end_ ~value:closed ~ranges:[closed] in
  let barrier = Uop.barrier ~srcs:[end_closed; end_closed] () in
  let value = Uop.O.(first + closed + last) in
  let root = Uop.after ~src:value ~deps:[barrier; barrier] in
  equal (list int) [Uop.tag first; Uop.tag last]
    (List.map Uop.tag (Uop.ranges root));
  is_true (Uop.ranges_subset root value);
  is_false (Uop.ranges_subset value root)

let range_order_and_membership_share_scope () =
  List.iter (fun membership_first ->
      let axis = if membership_first then 91300 else 91310 in
      let range i = Uop.range ~size:(Uop.const_int 4) ~axis:(axis + i)
          ~kind:Axis_type.Weak () in
      let first = range 0 and second = range 1 and third = range 2 in
      let left = Uop.O.(third + first) and right = Uop.O.(second + third) in
      let value = Uop.O.(left + right) in
      let nested = Uop.range ~size:(Uop.const_int 3) ~axis:(axis + 3)
          ~kind:Axis_type.Weak ~parents:[third; first; second; first] () in
      let closed = Uop.end_ ~value:nested ~ranges:[first; first; second] in
      let expected = [nested; third] in
      let membership () =
        List.iter (fun range ->
            equal bool (List.memq range expected) (Uop.ranges_subset range closed))
          [first; second; third];
        is_true (Uop.ranges_subset closed nested);
        is_false (Uop.ranges_subset nested closed) in
      if membership_first then membership ();
      equal (list int) (List.map Uop.tag [third; first; second])
        (List.map Uop.tag (Uop.ranges value));
      equal (list int) (List.map Uop.tag [nested; third; first; second])
        (List.map Uop.tag (Uop.ranges nested));
      equal (list int) (List.map Uop.tag expected)
        (List.map Uop.tag (Uop.ranges closed));
      membership ()) [false; true]

let linear_closes_ranges () =
  let r =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Weak ()
  in
  let lin = Uop.linear [ r ] in
  is_true ~msg:"Linear closes all of its source ranges"
    (not (List.exists (Uop.equal r) (Uop.ranges lin)))

(* Symbolic integers. [Uop.resolve]/[smax]/[smin]/[sprod]/[broadcast_shape]
   lean on the symbolic simplifier, which installs itself into
   [Uop.simplify_ref] when [Symbolic] is linked; force that linkage here. *)
let () =
  let (_ : Uop.t -> Uop.t) = Sys.opaque_identity Symbolic.simplify in
  ()

let resolve_decides_comparisons_from_bounds () =
  let v = Uop.variable ~name:"sint_r" ~min_val:1 ~max_val:10 () in
  is_true ~msg:"provably true comparison"
    (Uop.resolve Uop.O.(v < Uop.const_int 20));
  is_true ~msg:"provably false comparison"
    (not (Uop.resolve Uop.O.(Uop.const_int 20 < v)));
  let undecided = Uop.O.(v < Uop.const_int 5) in
  is_true ~msg:"undecidable defaults to true" (Uop.resolve undecided);
  is_true ~msg:"undecidable takes the given default"
    (not (Uop.resolve ~default:false undecided));
  let raises =
    try
      ignore (Uop.resolve (Uop.const_int 3));
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"non-boolean input rejected" raises

let smax_smin_fold_when_bounds_decide () =
  let v = Uop.variable ~name:"sint_m" ~min_val:3 ~max_val:10 () in
  is_true ~msg:"smax folds a dominated constant away"
    (Uop.equal (Uop.smax [ Uop.const_int 2; v ]) v);
  equal (option int) ~msg:"smax of constants folds" (Some 5)
    (Uop.const_int_value (Uop.smax [ Uop.const_int 2; Uop.const_int 5 ]));
  equal (option int) ~msg:"smin of constants folds" (Some 2)
    (Uop.const_int_value (Uop.smin [ Uop.const_int 5; Uop.const_int 2 ]));
  is_true ~msg:"smin folds a dominating constant away"
    (Uop.equal (Uop.smin [ Uop.const_int 2; v ]) (Uop.const_int 2));
  is_true ~msg:"smin keeps the dominated symbolic term"
    (Uop.equal (Uop.smin [ v; Uop.const_int 20 ]) v)

let sprod_simplifies () =
  equal (option int) ~msg:"empty product is one" (Some 1)
    (Uop.const_int_value (Uop.sprod []));
  equal (option int) ~msg:"constant product folds" (Some 6)
    (Uop.const_int_value (Uop.sprod [ Uop.const_int 2; Uop.const_int 3 ]));
  let v = Uop.variable ~name:"sint_p" ~min_val:1 ~max_val:4 () in
  is_true ~msg:"multiplication by one folds away"
    (Uop.equal (Uop.sprod [ Uop.const_int 1; v ]) v)

let broadcast_shape_symbolic_and_raising () =
  let v = Uop.variable ~name:"sint_b" ~min_val:0 ~max_val:6 () in
  let d = Uop.O.(v + Uop.const_int 1) in
  (match Uop.broadcast_shape [ [ Uop.const_int 1 ]; [ d ] ] with
   | [ out ] -> is_true ~msg:"one broadcasts against a symbolic dim"
       (Uop.equal out d)
   | _ -> fail "expected rank-1 broadcast");
  (match
     Uop.broadcast_shape
       [ [ Uop.const_int 4; Uop.const_int 1 ]; [ Uop.const_int 1; d ] ]
   with
   | [ a; b ] ->
       is_true ~msg:"aligned from the last axis"
         (Uop.equal a (Uop.const_int 4) && Uop.equal b d)
   | _ -> fail "expected rank-2 broadcast");
  (match Uop.broadcast_shape [ [ Uop.const_int 0 ]; [ Uop.const_int 1 ] ] with
   | [ z ] -> equal (option int) ~msg:"zero wins over one" (Some 0)
       (Uop.const_int_value z)
   | _ -> fail "expected rank-1 broadcast");
  let raises =
    try
      ignore (Uop.broadcast_shape [ [ Uop.const_int 3 ]; [ Uop.const_int 2 ] ]);
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"incompatible constants raise" raises;
  let w = Uop.variable ~name:"sint_b2" ~min_val:0 ~max_val:6 () in
  let raises_sym =
    try
      ignore (Uop.broadcast_shape [ [ d ]; [ Uop.O.(w + Uop.const_int 1) ] ]);
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"distinct symbolic dims raise" raises_sym

let inferred_broadcast_shapes_are_checked () =
  let param slot dims = Uop.param ~slot ~dtype:Dtype.float32
      ~shape:(Uop.stack dims) () in
  let sum a b = Uop.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let two = Uop.const_int 2 and three = Uop.const_int 3 in
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Uop.shape
      (sum (param 812 [ two ]) (param 813 [ three ]))));
  let n = Uop.variable ~name:"shape_n" ~min_val:1 ~max_val:8 () in
  let m = Uop.variable ~name:"shape_m" ~min_val:1 ~max_val:8 () in
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Uop.shape
      (sum (param 814 [ n ]) (param 815 [ m ]))));
  let same = sum (param 814 [ n ]) (param 816 [ n ]) in
  is_true (List.for_all2 Uop.equal [ n ] (Uop.shape same));
  let empty = sum (param 817 [ Uop.const_int 0 ])
      (param 818 [ Uop.const_int 1 ]) in
  equal (list int) [ 0 ] (List.map (fun d -> Option.get (Uop.const_int_value d))
      (Uop.shape empty))

let unbind_splits_bound_variables () =
  let v = Uop.variable ~name:"sint_u" ~min_val:0 ~max_val:10 () in
  let bound = Uop.bind ~var:v ~value:(Uop.const_int 7) in
  let var, value = Uop.unbind bound in
  is_true ~msg:"unbind returns the variable" (Uop.equal var v);
  equal int64 ~msg:"unbind returns the value" 7L value;
  let raises =
    try
      ignore (Uop.unbind v);
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"unbind of a plain variable raises" raises

let bind_validates_range () =
  let v = Uop.variable ~name:"sint_bd" ~min_val:2 ~max_val:5 () in
  ignore (Uop.bind ~var:v ~value:(Uop.const_int 2));
  ignore (Uop.bind ~var:v ~value:(Uop.const_int 5));
  let rejects value =
    try
      ignore (Uop.bind ~var:v ~value:(Uop.const_int value));
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"below-range bind value rejected" (rejects 1);
  is_true ~msg:"above-range bind value rejected" (rejects 6)

let compiled_signature_preserves_slots_and_types () =
  let module U = Uop in
  let extent = U.variable ~name:"extent" ~min_val:1 ~max_val:12 () in
  let b3 = U.param ~slot:3 ~name:"named_buffer" ~dtype:Dtype.float32
      ~shape:(U.stack [ extent; U.const_int 4 ]) () in
  let b11 = U.param ~slot:11 ~dtype:Dtype.int16 ~shape:(U.const_int 0) () in
  let v = U.variable ~param:true ~name:"value" ~min_val:(-10) ~max_val:10 ~dtype:Dtype.int64 () in
  let sink = U.sink [ b11; b3; v ] in
  let target = Target.of_string "PCI:2+AMD:HIP:gfx1100" in
  let info = { (U.program_info_from_sink ~target sink) with vars = [ v ] } in
  let program = U.program ~sink ~linear:(U.linear [ b11; v; U.buf_uop b3 ])
      ~source:(U.source "source") ~binary:(U.binary "\x7fELF\x00payload") ~info () in
  let obj = U.to_elf program in
  equal string "test" obj.name;
  equal string "\x7fELF\x00payload" (Bytes.to_string obj.lib);
  equal string "PCI:2+AMD:HIP:gfx1100" (Target.to_string obj.target);
  equal (option string) (Some (U.semantic_key program)) obj.profile_key;
  equal (list int) [ 1; 0; 2 ] (List.map (fun (a : Tiny_elf.argument) -> a.slot) obj.signature);
  equal (list (list int)) [ [ 0 ]; [ 48 ]; [] ]
    (List.map (fun (a : Tiny_elf.argument) -> a.shape) obj.signature);
  match obj.signature with
  | [ zero; named; scalar ] ->
      is_true (Dtype.equal zero.dtype Dtype.int16);
      equal (option string) (Some "named_buffer") named.name;
      is_true (named.addrspace = Dtype.Global);
      is_true (scalar.addrspace = Dtype.Alu);
      is_true (Dtype.equal scalar.dtype Dtype.int64)
  | _ -> fail "expected two buffers and one scalar"

let binary_argument_layout () =
  let arg slot addrspace dtype : Tiny_elf.argument =
    { name = None; slot; addrspace; dtype; shape = [] } in
  let signature = [ arg 0 Dtype.Global Dtype.float32;
    arg 1 Dtype.Alu Dtype.int8; arg 2 Dtype.Alu Dtype.int16;
    arg 3 Dtype.Alu Dtype.int32; arg 4 Dtype.Alu Dtype.int64 ] in
  let fields = Tiny_elf.layout signature in
  equal (list (pair int int)) [ 0, 8; 8, 1; 10, 2; 12, 4; 16, 8 ]
    (List.map (fun (field : Tiny_elf.field) -> field.offset, field.size) fields);
  equal (list int) [ 0; 1; 2; 3; 4 ]
    (List.map (fun (field : Tiny_elf.field) -> field.argument.slot) fields);
  List.iter (fun dtype ->
      raises_match (function Invalid_argument _ -> true | _ -> false)
        (fun () -> ignore (Tiny_elf.layout [ arg 0 Dtype.Alu dtype ])))
    [ Dtype.void; Dtype.weakint; Dtype.weakfloat ]

let incomplete_program_has_no_binary () =
  let sink = Uop.sink [] in
  let info = Uop.program_info_from_sink sink in
  List.iter (fun u ->
      raises_match (function Invalid_argument _ -> true | _ -> false)
        (fun () -> ignore (Uop.to_elf u)))
    [ sink; Uop.program ~sink ~info () ]

let commutative_axes_use_lexical_argument_order () =
  let axis n = Uop.range ~size:(Uop.const_int 32) ~axis:n ~kind:Axis_type.Weak () in
  let a = axis 2 and b = axis 10 in
  let sum = Uop.simplify Uop.O.(a + b) in
  is_true ~msg:"axis 10 sorts before axis 2" ((Uop.src sum).(0) == b && (Uop.src sum).(1) == a);
  is_true ~msg:"canonical order is idempotent" (Uop.simplify sum == sum)

let full_width_scalar_bindings () =
  let scalar = Uop.param ~slot:(-1) ~name:"full_width" ~dtype:Dtype.int64
      ~addrspace:Dtype.Alu
      ~vmin_vmax:(Dtype.min Dtype.int64, Dtype.max Dtype.int64) () in
  let variable = Uop.replace scalar ~op:Ops.Buffer () in
  let divisor = Uop.const (Const.int64 Dtype.weakint 0x4000_0000_0000_0000L) in
  let info = Uop.program_info_from_sink (Uop.sink [scalar]) in
  List.iter (fun (value, quotient) ->
      let binding = Uop.bind ~var:variable ~value:(Uop.const (Const.int64 Dtype.int64 value)) in
      equal int64 value (snd (Uop.unbind binding));
      equal (list int64) [value] (Uop.program_vals info ~var_vals:["full_width", value]);
      equal int quotient (Uop.sym_infer Uop.O.(scalar // divisor) ["full_width", value]);
      raises (Invalid_argument "sym_infer: result does not fit a host integer")
        (fun () -> Uop.sym_infer scalar ["full_width", value]))
    [Int64.min_int, -2; Int64.max_int, 1]

let constants_preserve_operand_shape () =
  let scalar = Uop.variable ~name:"constant_scalar" ~min_val:0 ~max_val:9
      ~dtype:Dtype.int32 () in
  let vector = Uop.stack [scalar; scalar; scalar] in
  let rows = Uop.variable ~name:"constant_rows" ~min_val:1 ~max_val:4 () in
  let tensor = Uop.param ~slot:95301 ~dtype:Dtype.float32
      ~shape:(Uop.stack [rows; Uop.const_int 3]) () in
  List.iter (fun source ->
      List.iter (fun (value, constant) ->
          is_true ~msg:"constant keeps the exact symbolic shape"
            (List.equal Uop.equal (Uop.shape source) (Uop.shape constant));
          is_true ~msg:"constant keeps the requested scalar dtype"
            (Dtype.equal (Uop.dtype source) (Uop.dtype constant));
          is_true ~msg:"constant keeps its value"
            (Bound.equal (Uop.vmin constant) (Bound.int value)
             && Bound.equal (Uop.vmax constant) (Bound.int value)))
        [0, Uop.zero_like source; 7, Uop.const_like source 7])
    [scalar; vector; tensor]

let () =
  run "tolk.uop"
    [
      group "Construction"
        [
          test "full-width scalar bindings retain exact values and checked launch arithmetic"
            full_width_scalar_bindings;
          test "constant-like values preserve scalar, vector and symbolic tensor shapes"
            constants_preserve_operand_shape;
          test "compiled signatures preserve sparse slots and argument types"
            compiled_signature_preserves_slots_and_types;
          test "argument structures align pointers and mixed scalar widths" binary_argument_layout;
          test "incomplete programs have no binary" incomplete_program_has_no_binary;
          test "weak axes follow lexical argument order" commutative_axes_use_lexical_argument_order;
          test "Ops and dtype access" ops_access;
          test "Ops tinygrad order" ops_tinygrad_order;
          test "Ops.Group algebra" group_algebra;
          test "hash-consing yields same node" hashcons_identity;
          test "Add has two srcs" add_has_two_srcs;
          test "infix O module builds Mul" infix_builds_mul;
          test "usum and uprod fold booleans with Or and And"
            bool_folds_are_logical;
          test "scalar BUFFER carries Param_arg" param_arg_symbolic_constructor;
          test "tinygrad arithmetic helper parity"
            arithmetic_helpers_tinygrad_parity;
          test "binding requires a concrete value" bind_requires_concrete_value;
          test "tinygrad integer bounds parity" integer_bounds_parity;
          test "integer bounds wrap at a fixed width" wrapping_integer_bounds;
          test "tinygrad CAST bounds parity" cast_bounds_parity;
          test "flat storage parameters retain symbolic views" flat_storage_parameters;
          test "backward slices track shared dependencies" backward_slice_tracks_shared_dependencies;
          test "allocations preserve shape and address space" allocations_preserve_shape_and_address_space;
          test "placeholder checks shape product" placeholder_checks_shape_product;
          test "max_numel checks host range" max_numel_checks_host_range;
          test "max_numel handles zero after large dimensions"
            max_numel_handles_zero_after_large_dimensions;
          test "movement dimensions do not wrap" movement_dimensions_do_not_wrap;
          test "bitcast dimensions remain exact until host conversion"
            bitcast_dimensions_remain_exact_until_host_conversion;
          test "exact symbolic bounds" exact_symbolic_bounds;
          test "Stack/Stage constructors"
            stack_stage_slice_constructors;
          test "UOp constructor parity shortcuts"
            uop_constructor_parity_shortcuts;
          test "scalar constant payload constructors"
            const_scalar_payload_constructors;
          test "tinygrad call constructor parity"
            call_constructor_parity;
          test "tinygrad property helper parity" property_helpers_parity;
          test "division promotes integer operands" division_promotes_integer_operands;
          test "STACK promotes all operands" stack_promotes_all_operands;
          test "tinygrad runtime realization state parity"
            runtime_realization_state_parity;
          test "Reduce layouts" reduce_layouts;
          test "BINARY and GETADDR dtypes" binary_and_getaddr_dtypes;
          test "void and value op shapes" void_and_value_op_shapes;
          test "STACK prepends a leading dim" stack_prepends_leading_dim;
          test "EXPAND prepends leading dims" prepend_expand;
          test "BITCAST size change" bitcast_size_change;
          test "child_ops reports the child op set"
            child_ops_reports_child_op_set;
          test "property caches release nodes" property_caches_release_nodes;
          test "exec_alu folds and absorbs invalids"
            exec_alu_folds_and_absorbs;
          test "exec_alu division preserves IEEE exceptional values" exec_alu_float_division;
          test "float-to-weak-int conversion retains all bits" scalar_float_to_weak_integer;
          test "exec_alu handles unsigned and float/integer boundaries" scalar_width_boundaries;
          test "exec_alu preserves exact scalar values" exec_alu_exact_scalars;
          test "exec_alu keeps unbounded weak intermediates" exec_alu_weak_intermediates;
          test "exec_alu TRUNC keeps non-finite inputs"
            exec_alu_trunc_keeps_nonfinite;
          test "alu_unary promotes transcendentals"
            alu_unary_promotes_transcendentals;
          test "semantic tag and side metadata"
            semantic_tag_and_side_metadata;
          test "info function name parity"
            info_function_names_follow_tinygrad;
          test "cache info semantic_key parity"
            cache_info_semantic_key_parity;
          test "remove_all_tags parity" remove_all_tags_parity;
          test "Program constructor prefix layouts"
            program_constructor_prefix_layouts;
          test "ProgramInfo.from_sink parity" program_info_from_sink_parity;
          test "ProgramInfo launch floor div/mod"
            program_launch_dims_floor_divmod;
          test "symbolic inference uses exact host scalar semantics"
            sym_infer_host_scalars;
          test "tinygrad-shaped debug toposort"
            debug_prints_toposort_like_tinygrad;
          test "tinygrad-shaped debug ranges and supplied lists"
            debug_prints_ranges_and_supplied_list_sources;
          test "tinygrad debug dtype reprs"
            debug_prints_tinygrad_dtype_reprs;
          test "tinygrad debug float and special args"
            debug_prints_float_and_special_args_like_tinygrad;
          test "tinygrad debug direct string args"
            debug_prints_direct_string_args_like_tinygrad;
          test "tinygrad debug range ordering"
            debug_prints_ranges_in_tinygrad_arg_order;
          test "tinygrad debug rich arg reprs"
            debug_prints_rich_args_dataclass_style;
          test "tinygrad debug reduce arg repr"
            debug_prints_reduce_arg_tuple;
          test "debug output ignores side metadata"
            debug_print_ignores_side_metadata;
          test "debug listing omits tags" debug_listing_omits_tags;
        ];
      group "Upat"
        [
          test "matches an Add node" upat_matches_add;
          test "captures named operands" upat_captures_operands;
          test "Pattern_matcher rewrites identities" pattern_matcher_rewrites;
          test "operator surface matches tinygrad"
            upat_operator_surface_matches_tinygrad;
          test "Pattern_matcher threads context"
            pattern_matcher_context_rewrites;
          test "matches node tags" upat_matches_node_tags;
          test "numeric literals compare exactly" upat_numeric_literals;
          test "Pattern_matcher rejects op-less rules"
            pattern_matcher_rejects_opless_rules;
          test "context matcher rejects op-less rules"
            context_matcher_rejects_opless_rules;
          test "custom early reject skips callback"
            custom_early_reject_skips_callback;
          test "dtype matches scalar of vector"
            upat_dtype_matches_scalar_of_vector;
          test "explicit source pattern helpers" upat_explicit_source_patterns;
          test "dedicated reduce arg predicates" upat_matches_reduce_arg;
          test "reserved ctx capture is rejected"
            upat_rejects_reserved_ctx_capture;
          test "permutation matches are deduplicated"
            upat_permutation_matches_are_deduplicated;
          test "self replacement is ignored"
            pattern_matcher_ignores_self_replacement;
          test "walk does not enter replacement subtrees"
            graph_rewrite_walk_does_not_enter_replacements;
          test "graph rewrite bpm and post ordering"
            graph_rewrite_bpm_runs_before_post_order;
          test "walk bpm short-circuits replacement subtrees"
            graph_rewrite_walk_bpm_short_circuits_replacement;
          test "Bottom_up_gate skips post and children"
            graph_rewrite_bottom_up_gate_skips_post_and_children;
          test "walk Bottom_up_gate skips post and children"
            graph_rewrite_walk_bottom_up_gate_skips_post_and_children;
          test "graph rewrite enters native callee expressions"
            graph_rewrite_enters_native_callee;
          test "graph rewrite skips call bodies by default"
            graph_rewrite_skips_call_body_by_default;
          test "graph rewrite pins call bodies on every path"
            graph_rewrite_pins_call_body_on_every_path;
          test "graph rewrite detects bottom-up cycles"
            graph_rewrite_detects_bottom_up_cycles;
          test "graph rewrite detects top-down cycles"
            graph_rewrite_detects_top_down_cycles;
        ];
      group "Ranges"
        [
          test "shared ending dependencies have bounded allocations"
            shared_ending_dependencies_have_bounded_allocations;
          test "nested ending dependencies preserve live range order"
            nested_ending_dependencies_preserve_live_range_order;
          test "range order and membership share scope in either query order"
            range_order_and_membership_share_scope;
          test "After closes dependency ranges"
            after_closes_ranges_from_dependencies;
          test "Linear closes ranges" linear_closes_ranges;
        ];
      group "Symbolic integers"
        [
          test "resolve decides comparisons from bounds"
            resolve_decides_comparisons_from_bounds;
          test "smax/smin fold when bounds decide"
            smax_smin_fold_when_bounds_decide;
          test "sprod simplifies" sprod_simplifies;
          test "Inferred broadcast shapes are checked" inferred_broadcast_shapes_are_checked;
          test "broadcast_shape handles symbolic dims and raises"
            broadcast_shape_symbolic_and_raising;
          test "unbind splits bound variables"
            unbind_splits_bound_variables;
          test "bind validates the variable range" bind_validates_range;
        ];
    ]
