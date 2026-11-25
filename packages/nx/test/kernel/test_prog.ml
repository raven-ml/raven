(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Programs as prog.mli states them: the kinds' domains against its table,
   constants' bits against an array's bytes, programs against what they were
   made of, expressions against a depth-first walk that shares by structure, and
   ranges against the values a program computes on nx.cpu. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog

let strf = Printf.sprintf
let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let name (D.Any dt) = D.name dt
let f32 = D.Any D.Float32
let f64 = D.Any D.Float64
let i32 = D.Any D.Int32
let i64 = D.Any D.Int64
let boolean = D.Any D.Bool
let bit = D.Any D.Bit

(* Kinds and their names *)

(* Each kind with its name in messages. *)
let unaries =
  P.
    [
      (Neg, "neg");
      (Recip, "recip");
      (Abs, "abs");
      (Sign, "sign");
      (Sqrt, "sqrt");
      (Exp, "exp");
      (Exp2, "exp2");
      (Log, "log");
      (Log2, "log2");
      (Log1p, "log1p");
      (Expm1, "expm1");
      (Sin, "sin");
      (Cos, "cos");
      (Tan, "tan");
      (Asin, "asin");
      (Acos, "acos");
      (Atan, "atan");
      (Sinh, "sinh");
      (Cosh, "cosh");
      (Tanh, "tanh");
      (Asinh, "asinh");
      (Acosh, "acosh");
      (Atanh, "atanh");
      (Erf, "erf");
      (Floor, "floor");
      (Ceil, "ceil");
      (Round, "round");
      (Trunc, "trunc");
    ]

let binaries =
  P.
    [
      (Add, "add");
      (Sub, "sub");
      (Mul, "mul");
      (Fdiv, "fdiv");
      (Idiv, "idiv");
      (Mod, "mod");
      (Pow, "pow");
      (Atan2, "atan2");
      (Maximum, "maximum");
      (Minimum, "minimum");
      (And, "and");
      (Or, "or");
      (Xor, "xor");
      (Threefry, "threefry");
    ]

let compares =
  P.
    [
      (Equal, "equal");
      (Not_equal, "not_equal");
      (Less, "less");
      (Less_equal, "less_equal");
    ]

let unary_name u = List.assoc u unaries
let binary_name b = List.assoc b binaries
let compare_name c = List.assoc c compares

let op1s =
  P.[ Copy; Cast; Bitcast ] @ List.map (fun (u, _) -> P.Unary u) unaries

let op2s =
  List.map (fun (b, _) -> P.Binary b) binaries
  @ List.map (fun (c, _) -> P.Compare c) compares

let op3s = P.[ Where; Fma ]

let op1_name = function
  | P.Copy -> "copy"
  | Cast -> "cast"
  | Bitcast -> "bitcast"
  | Unary u -> unary_name u

let op2_name = function
  | P.Binary b -> binary_name b
  | Compare c -> compare_name c

let op3_name = function P.Where -> "where" | Fma -> "fma"

(* A kind of any arity, for the tables over every kind. *)
type kind = K1 of P.op1 | K2 of P.op2 | K3 of P.op3

let kinds =
  List.map (fun k -> K1 k) op1s
  @ List.map (fun k -> K2 k) op2s
  @ List.map (fun k -> K3 k) op3s

let kind_name = function
  | K1 k -> op1_name k
  | K2 k -> op2_name k
  | K3 k -> op3_name k

(* Nodes and programs *)

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i -> strf "%02x" (Char.code s.[i])))

let pp_node ppf = function
  | P.In k -> Format.fprintf ppf "In %d" k
  | Coord k -> Format.fprintf ppf "Coord %d" k
  | Const (d, s) -> Format.fprintf ppf "Const (%s, %s)" (name d) (hex s)
  | Op1 (k, d, a) ->
      Format.fprintf ppf "Op1 (%s, %s, %d)" (op1_name k) (name d) a
  | Op2 (k, a, b) -> Format.fprintf ppf "Op2 (%s, %d, %d)" (op2_name k) a b
  | Op3 (k, a, b, c) ->
      Format.fprintf ppf "Op3 (%s, %d, %d, %d)" (op3_name k) a b c

let node = Testable.make ~pp:pp_node ~equal:( = )

let pp_parts ppf (ins, nodes, outs) =
  let list pp ppf xs =
    let sep ppf () = Format.fprintf ppf ";@ " in
    Format.pp_print_list ~pp_sep:sep pp ppf xs
  in
  let dtype ppf d = Format.pp_print_string ppf (name d) in
  Format.fprintf ppf
    "@[<hv 2>{ ins = [%a];@ nodes = [@[<hov>%a@]];@ outs = [%a] }@]"
    (list dtype) ins (list pp_node) nodes (list Format.pp_print_int) outs

let pp_prog ppf (p : P.t) =
  pp_parts ppf
    (Iarray.to_list p.ins, Iarray.to_list p.nodes, Iarray.to_list p.outs)

let structural = Testable.structural ~pp:pp_prog

(* A program's parts as plain arrays, to draw and to rebuild. *)
type drawn = { ins : D.any array; nodes : P.node array; outs : int array }

let pp_drawn ppf d =
  pp_parts ppf (Array.to_list d.ins, Array.to_list d.nodes, Array.to_list d.outs)

let program d =
  P.v ~ins:(Iarray.of_array d.ins) (Iarray.of_array d.nodes)
    ~outs:(Iarray.of_array d.outs)

let drawn_of (p : P.t) =
  {
    ins = Iarray.to_array p.ins;
    nodes = Iarray.to_array p.nodes;
    outs = Iarray.to_array p.outs;
  }

(* The element of [dt] whose bits are [b]. *)
let element (type v s) (dt : (v, s) D.t) b : v =
  A.get (A.v dt (Nx_array.Layout.contiguous [||]) (Rig.Buffer.of_string b)) [||]

(* The minor words one call of [f] allocates, over 1000 calls after a first. *)
let words f =
  ignore (f ());
  let n = 1000 in
  let before = Gc.minor_words () in
  for _ = 1 to n do
    ignore (Sys.opaque_identity (f ()))
  done;
  (Gc.minor_words () -. before) /. Float.of_int n

(* [p]'s outputs over [loads] at [shape], computed by nx.cpu, as their elements'
   bits. *)
let compute ?(shape = [| 1 |]) (p : P.t) loads =
  let dst o =
    let (D.Any dt) = P.dtype p o in
    A.Any (A.create Rig.host dt (Iarray.of_array shape))
  in
  let dsts = Iarray.map dst p.outs in
  let loads =
    Iarray.map (fun (A.Any x) -> Nx_kernel.Plain x) (Iarray.of_array loads)
  in
  match Nx_cpu.map p ~dsts loads with
  | Done -> Ok (Array.map Elements.elements (Iarray.to_array dsts))
  | answer -> Error answer

let outputs = Testable.(result (array (array string)) Elements.answer)

(* A one-element operand of [d] whose bits are [b]. *)
let operand1 d b = Elements.of_elements d [| 1 |] [| b |]

(* Drawn programs *)

let base_dtypes =
  D.
    [
      Any Float32;
      Any Float64;
      Any Float16;
      Any Int32;
      Any Uint8;
      Any Int64;
      Any Uint64;
      Any Bool;
      Any Complex64;
      Any Int4;
    ]

let same (D.Any x) (D.Any y) = D.equal x y

(* Bits that dtypes of one width share: [-0.], three NaN payloads and [1.] at
   float32 and float64, also an int32, int64 or uint64. *)
let shared_bits =
  let four v =
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 v;
    Bytes.to_string b
  in
  let eight v =
    let b = Bytes.create 8 in
    Bytes.set_int64_le b 0 v;
    Bytes.to_string b
  in
  List.map four
    [ 0x8000_0000l; 0x7fc0_0001l; 0x7fc0_0002l; 0xffc0_0001l; 0x3f80_0000l ]
  @ List.map eight
      [
        0x8000_0000_0000_0000L;
        0x7ff8_0000_0000_0001L;
        0x7ff8_0000_0000_0002L;
        0xfff8_0000_0000_0001L;
        0x3ff0_0000_0000_0000L;
      ]

(* Whether [d]'s elements hold the bits [b] as they are. *)
let holds_bits (D.Any dt) b =
  String.length b = D.bytes dt 1 && P.bits dt (element dt b) = b

(* The constants of [d] a program draws: [-0.] and the integers 0 to 6, and the
   shared bits [d] holds. *)
let constants (D.Any dt as d) =
  let ints =
    List.map
      (fun x -> P.bits dt (D.of_float dt x))
      [ -0.; 0.; 1.; 2.; 3.; 4.; 5.; 6. ]
  in
  List.map
    (fun b -> P.Const (d, b))
    (ints @ List.filter (holds_bits d) shared_bits)

(* A program of [n] nodes over operands [ins], each node read from [choices]: an
   operand, an axis or a constant, a copy of an earlier node, or a kind over
   earlier nodes of the dtypes it takes. Each kind's first operand is the last
   kind made, so that the last node reaches most of the program. Half the
   constants are shared bits in a dtype of their width. Outputs repeat and leave
   nodes unreached. *)
let grow ins choices n nouts =
  let next = ref 0 in
  let int k =
    let c = choices.(!next mod Array.length choices) in
    incr next;
    c mod k
  in
  let pick l = List.nth l (int (List.length l)) in
  let types = Array.make n i64 and nodes = Array.make n (P.Coord 0) in
  (* Earlier nodes of [i]'s dtype, [i] among them. *)
  let alike i k =
    List.filter (fun j -> same types.(j) types.(i)) (List.init k Fun.id)
  in
  let constant () =
    match int 2 with
    | 0 ->
        let d = pick base_dtypes in
        (pick (constants d), d)
    | _ ->
        let b = pick shared_bits in
        let d = pick (List.filter (fun d -> holds_bits d b) base_dtypes) in
        (P.Const (d, b), d)
  in
  let leaf () =
    match int 3 with
    | 0 when Array.length ins > 0 ->
        let k = int (Array.length ins) in
        (P.In k, ins.(k))
    | 1 -> (P.Coord (int 3), i64)
    | _ -> constant ()
  in
  let copy i = (P.Op1 (Copy, types.(i), i), types.(i)) in
  let op k i =
    let (D.Any x) = types.(i) in
    match int 5 with
    | 0 -> (nodes.(i), types.(i))
    | 1 -> (
        let (D.Any y as d) = pick base_dtypes in
        match List.filter (fun o -> P.accepts1 o x y) op1s with
        | [] -> copy i
        | os -> (P.Op1 (pick os, d, i), d))
    | 2 -> (
        let j = pick (alike i k) in
        match List.filter (fun o -> P.accepts2 o x) op2s with
        | [] -> copy i
        | os ->
            let o = pick os in
            let d =
              match o with Compare _ -> boolean | Binary _ -> types.(i)
            in
            (P.Op2 (o, i, j), d))
    | 3 when D.is D.Boolean x ->
        let j = int k in
        let l = if int 2 = 0 then j else pick (alike j k) in
        (P.Op3 (Where, i, j, l), types.(j))
    | _ when P.accepts3 Fma x x ->
        let j = pick (alike i k) and l = pick (alike i k) in
        let a, b, c =
          match int 5 with
          | 0 -> (i, i, i)
          | 1 -> (i, i, j)
          | 2 -> (i, j, i)
          | 3 -> (j, i, i)
          | _ -> (i, j, l)
        in
        (P.Op3 (Fma, a, b, c), types.(i))
    | _ -> copy i
  in
  let last = ref (-1) in
  for k = 0 to n - 1 do
    let nd, dt =
      if k = 0 || int 5 = 0 then leaf ()
      else
        let i = if !last >= 0 then !last else int k in
        last := k;
        op k i
    in
    nodes.(k) <- nd;
    types.(k) <- dt
  done;
  let outs = Array.init nouts (fun _ -> if int 2 = 0 then n - 1 else int n) in
  { ins; nodes; outs }

(* Programs of up to 400 nodes over [ins_size] operands, so that of_exprs's
   lowering scans small programs and crosses to tables past 40 nodes. *)
let drawn_over ins_size =
  let open Gen in
  with_pp pp_drawn
    (let* n =
       frequency
         [ (2, int_range 1 8); (1, int_range 50 100); (1, int_range 170 400) ]
     in
     let+ ins = array ~size:ins_size (of_list ~pp:pp_dtype base_dtypes)
     and+ choices = array ~size:(constant (4 * n)) (int_range 0 0x3FFF_FFFF)
     and+ nouts = int_range 1 4 in
     grow ins choices n nouts)

let drawn = drawn_over (Gen.int_range 0 3)

(* Expressions of drawn programs *)

type ex = Ex : ('v, 's) P.E.t -> ex

let any (Ex x) = P.E.any x

(* [y] at [x]'s dtype, which a drawn node gives them both. *)
let like (type v s w r) (x : (v, s) P.E.t) (y : (w, r) P.E.t) : (v, s) P.E.t =
  match D.equal_witness (P.E.dtype y) (P.E.dtype x) with
  | Some Equal -> y
  | None -> failwith "a drawn node's operands differ in dtype"

(* [d]'s nodes as expressions, each over its operands' expressions, so that a
   node read twice is one value. A leaf [n] is [e] where [leaf n] is [Some
   e]. *)
let typed ?(leaf = fun _ -> None) d =
  let e = Array.make (Array.length d.nodes) (Ex (P.E.coord 0)) in
  let leaf_of = function
    | (P.In _ | Coord _ | Const _) as nd -> leaf nd
    | Op1 _ | Op2 _ | Op3 _ -> None
  in
  Array.iteri
    (fun i nd ->
      e.(i) <-
        (match (nd, leaf_of nd) with
        | _, Some x -> x
        | P.In k, None ->
            let (D.Any dt) = d.ins.(k) in
            Ex (P.E.input k dt)
        | Coord a, None -> Ex (P.E.coord a)
        | Const (D.Any dt, b), None -> Ex (P.E.const dt (element dt b))
        | Op1 (k, D.Any dt, a), None ->
            let (Ex x) = e.(a) in
            Ex (P.E.op1 k dt x)
        | Op2 (k, a, b), None -> (
            let (Ex x) = e.(a) in
            let (Ex y) = e.(b) in
            match k with
            | Binary k -> Ex (P.E.binary k x (like x y))
            | Compare k -> Ex (P.E.compare k x (like x y)))
        | Op3 (k, a, b, c), None ->
            let (Ex c') = e.(a) in
            let (Ex x) = e.(b) in
            let (Ex y) = e.(c) in
            Ex (P.E.op3 k c' x (like x y))))
    d.nodes;
  e

(* of_exprs's nodes and outputs as prog.mli states them for [d]'s outputs: the
   nodes they reach, each structure once, in the order a depth-first walk from
   the outputs, operands left to right, finishes them. *)
let lowered d =
  let index = Hashtbl.create 16 and nodes = ref [] and count = ref 0 in
  let seen = Array.make (Array.length d.nodes) (-1) in
  let rec visit i =
    if seen.(i) < 0 then begin
      let nd =
        match d.nodes.(i) with
        | (P.In _ | Coord _ | Const _) as nd -> nd
        | Op1 (k, dt, a) -> Op1 (k, dt, visit a)
        | Op2 (k, a, b) ->
            let a = visit a in
            Op2 (k, a, visit b)
        | Op3 (k, a, b, c) ->
            let a = visit a in
            let b = visit b in
            Op3 (k, a, b, visit c)
      in
      seen.(i) <-
        (match Hashtbl.find_opt index nd with
        | Some n -> n
        | None ->
            Hashtbl.add index nd !count;
            nodes := nd :: !nodes;
            incr count;
            !count - 1)
    end;
    seen.(i)
  in
  let outs = Array.map visit d.outs in
  (Array.of_list (List.rev !nodes), outs)

(* Whether each node of [d] is one the outputs [outs] reach. *)
let reach d outs =
  let seen = Array.make (Array.length d.nodes) false in
  let rec visit i =
    if not seen.(i) then begin
      seen.(i) <- true;
      match d.nodes.(i) with
      | P.In _ | Coord _ | Const _ -> ()
      | Op1 (_, _, a) -> visit a
      | Op2 (_, a, b) ->
          visit a;
          visit b
      | Op3 (_, a, b, c) ->
          visit a;
          visit b;
          visit c
    end
  in
  Array.iter visit outs;
  seen

(* Each axis's offset, all zero one draw in three. *)
let offsets =
  let open Gen in
  frequency
    [
      (1, constant [| 0; 0; 0 |]); (2, array ~size:(constant 3) (int_range 0 3));
    ]

(* The substitution adding [off] to each coordinate. *)
let shift off : P.node -> ex option = function
  | Coord c when off.(c) <> 0 ->
      Some (Ex P.E.(coord c + const D.Int64 (Int64.of_int off.(c))))
  | _ -> None

(* subst of [d]'s program over [ins], each leaf [n] replaced by [f n]'s
   expression where it is [Some]. *)
let subst ?outs ~ins f d =
  P.subst ?outs ~ins:(Iarray.of_array ins)
    (fun n -> Option.map any (f n))
    (program d)

(* Programs *)

let test_box_muller () =
  let f = D.Float32 in
  let lit x = P.E.const f x in
  let u1 = P.E.input 0 f and u2 = P.E.input 1 f in
  let z = P.E.(sqrt (lit (-2.) * log u1) * cos (lit (2. *. Float.pi) * u2)) in
  let p = P.of_exprs ~ins:[| f32; f32 |] [| P.E.any z |] in
  let const x = P.Const (f32, P.bits f x) in
  (* The walk from z: sqrt's product, -2 then log u1, then cos's product, 2 pi
     then u2, then z. *)
  let want : P.node array =
    [|
      const (-2.);
      In 0;
      Op1 (Unary Log, f32, 1);
      Op2 (Binary Mul, 0, 2);
      Op1 (Unary Sqrt, f32, 3);
      const (2. *. Float.pi);
      In 1;
      Op2 (Binary Mul, 5, 6);
      Op1 (Unary Cos, f32, 7);
      Op2 (Binary Mul, 4, 8);
    |]
  in
  equal ~msg:"nodes" (array node) want (Iarray.to_array p.nodes);
  equal ~msg:"outs" (array int) [| 9 |] (Iarray.to_array p.outs)

let law_equal_programs d =
  let p = program d and q = program d in
  equal structural p q;
  equal ~msg:"hashes" int (Hashtbl.hash p) (Hashtbl.hash q)

(* Domains *)

(* prog.mli's domains, by kind of number: "Neg, Recip, Sign, Add, Sub, Mul, Pow:
   every dtype but booleans. Fdiv: floats and complex. Idiv: integers. Abs,
   Floor, Ceil, Round, Trunc, Mod, Fma: floats and integers. Sqrt, Exp, Exp2,
   Log, Log2, and Sin to Tanh: floats and complex. Asinh, Acosh, Atanh: complex.
   Log1p, Expm1, Erf, Atan2: floats. And, Or, Xor: integers and booleans.
   Threefry: uint64. Maximum, Minimum and the comparisons: every dtype. Copy:
   every dtype, into itself. Cast: every pair. Bitcast: every pair of one width.
   Where: a boolean, then two operands of any one dtype." *)
type cls = Float | Signed | Unsigned | Boolean | Complex

let cls (D.Any dt) : cls =
  match D.kind dt with
  | Float -> Float
  | Signed -> Signed
  | Unsigned -> Unsigned
  | Boolean -> Boolean
  | Complex -> Complex

let integers = [ Signed; Unsigned ]
let reals = Float :: integers
let numbers = Complex :: reals
let every = Boolean :: numbers

let unary_domain : P.unary -> cls list = function
  | Neg | Recip | Sign -> numbers
  | Abs | Floor | Ceil | Round | Trunc -> reals
  | Sqrt | Exp | Exp2 | Log | Log2 | Sin | Cos | Tan | Asin | Acos | Atan | Sinh
  | Cosh | Tanh ->
      [ Float; Complex ]
  | Log1p | Expm1 | Erf -> [ Float ]
  | Asinh | Acosh | Atanh -> [ Complex ]

let binary_domain : P.binary -> cls list = function
  | Add | Sub | Mul | Pow -> numbers
  | Fdiv -> [ Float; Complex ]
  | Idiv -> integers
  | Mod -> reals
  | Atan2 -> [ Float ]
  | Maximum | Minimum -> every
  | And | Or | Xor -> Boolean :: integers
  | Threefry -> []

let width (D.Any dt) = D.bits dt

let in_domain1 k x y =
  match k with
  | P.Copy -> same x y
  | Cast -> true
  | Bitcast -> width x = width y
  | Unary u -> same x y && List.mem (cls x) (unary_domain u)

let in_domain2 k x =
  match k with
  | P.Binary Threefry -> name x = "uint64"
  | Binary b -> List.mem (cls x) (binary_domain b)
  | Compare _ -> true

let in_domain3 k c x =
  match k with
  | P.Where -> cls c = Boolean
  | Fma -> same c x && List.mem (cls x) reals

let names_where f = List.map name (List.filter f D.all)

let test_domain = function
  | K1 k ->
      List.iter
        (fun (D.Any x as ax) ->
          equal ~msg:(name ax) (list string)
            (names_where (in_domain1 k ax))
            (names_where (fun (D.Any y) -> P.accepts1 k x y)))
        D.all
  | K2 k ->
      equal (list string)
        (names_where (in_domain2 k))
        (names_where (fun (D.Any x) -> P.accepts2 k x))
  | K3 k ->
      List.iter
        (fun (D.Any c as ac) ->
          equal ~msg:(name ac) (list string)
            (names_where (in_domain3 k ac))
            (names_where (fun (D.Any x) -> P.accepts3 k c x)))
        D.all

(* Bits *)

(* An element's bits as a one-element array holds them: its bytes, or a sub-byte
   code in one byte. *)
let array_bits (type v s) (dt : (v, s) D.t) (x : v) =
  let a = A.of_array dt [| 1 |] [| x |] in
  match D.bits dt with
  | 1 ->
      String.make 1
        (Char.chr (Bool.to_int (A.get (A.expect D.Bit (A.Any a)) [| 0 |])))
  | 4 ->
      String.make 1
        (Char.chr (A.get (Option.get (A.bitcast D.Uint4 a)) [| 0 |]))
  | _ ->
      let b = A.to_array (Option.get (A.bitcast D.Uint8 a)) in
      String.init (Array.length b) (fun i -> Char.chr b.(i))

(* The value of [dt] that the words [w] and [w'] give: [w] as a double for a
   float, [w] and [w'] as the parts of a complex number, [w]'s low bits for an
   integer, signed or not, and [w <> 0] for a boolean. *)
let value_of_words (type v s) (dt : (v, s) D.t) w w' : v =
  let low n = Int64.to_int w land ((1 lsl n) - 1) in
  let signed n =
    let u = low n in
    if u >= 1 lsl (n - 1) then u - (1 lsl n) else u
  in
  match (D.kind dt, dt) with
  | Float, _ -> Int64.float_of_bits w
  | Complex, _ ->
      { Complex.re = Int64.float_of_bits w; im = Int64.float_of_bits w' }
  | Boolean, _ -> w <> 0L
  | _, D.Int64 -> w
  | _, D.Uint64 -> w
  | _, D.Int32 -> Int64.to_int32 w
  | _, D.Uint32 -> Int64.to_int32 w
  | _, D.Int16 -> signed 16
  | _, D.Uint16 -> low 16
  | _, D.Int8 -> signed 8
  | _, D.Uint8 -> low 8
  | _, D.Int4 -> signed 4
  | _, D.Uint4 -> low 4
  | _ -> invalid_arg (D.name dt)

(* The float formats a store rounds into, as each format's facts. *)
let rounded_formats =
  [
    ("float32", D.float_format D.Float32);
    ("float16", D.float_format D.Float16);
    ("bfloat16", D.float_format D.Bfloat16);
    ("float8_e4m3fn", D.float_format D.Float8_e4m3fn);
    ("float8_e5m2", D.float_format D.Float8_e5m2);
    ("float4_e2m1fn", D.float_format D.Float4_e2m1fn);
  ]

(* A double halfway between two neighbours of format [f] in a binade from its
   least normal value to its largest, of either sign: [(2m + 1) 2^(e - p)], [m]
   of [p] bits, [p] the format's significand. *)
let tie (f : D.float_format) =
  let open Gen in
  let p = f.fraction_bits + 1 in
  let _, lo = Float.frexp f.min_normal and _, hi = Float.frexp f.max_finite in
  let+ e = int_range (lo - 1) (hi - 1)
  and+ m = int_range (1 lsl (p - 1)) ((1 lsl p) - 1)
  and+ negative = bool in
  let x = Float.ldexp (Float.of_int ((2 * m) + 1)) (e - p) in
  Int64.bits_of_float (if negative then -.x else x)

(* Whether [x] is halfway between two neighbours of format [f]: within its
   normal values and of one more significant bit than [f] holds, the last
   set. *)
let is_tie (f : D.float_format) x =
  let m, _ = Float.frexp x in
  let s = Float.ldexp m (f.fraction_bits + 2) in
  Float.is_finite x
  && Float.abs x >= f.min_normal
  && Float.abs x <= f.max_finite
  && Float.is_integer s
  && Float.rem s 2. <> 0.

let word =
  let open Gen in
  frequency
    [
      (1, int64);
      (1, map Int64.bits_of_float float);
      (6, one_of (List.map (fun (_, f) -> tie f) rounded_formats));
    ]

(* Words a store treats apart: signed zeros, infinities, quiet and signalling
   NaNs of both signs, the extremes of a double, the largest finite values of
   the narrow floats and the doubles past them that round or saturate, the least
   subnormals and the halves below them, and every integer width's extremes in
   the low bits. *)
let special_words =
  List.map Int64.bits_of_float
    [
      0.;
      -0.;
      infinity;
      neg_infinity;
      Float.max_float;
      65504.;
      65519.;
      65520.;
      448.;
      464.;
      480.;
      57344.;
      61440.;
      6.;
      7.;
      0x1p-24;
      0x1p-25;
      0x1p-149;
      0x1p-150;
      0x1.fe0000000001p127;
      0x1.ffffffp127;
    ]
  @ [
      0x7FF8000000000001L;
      0x7FF0000000000001L;
      0xFFF4000000000000L;
      -1L;
      1L;
      0x7L;
      0x8L;
      0xFL;
      0x7FL;
      0x80L;
      0xFFL;
      0x7FFFL;
      0x8000L;
      0xFFFFL;
      0x7FFFFFFFL;
      0x80000000L;
      0xFFFFFFFFL;
      Int64.max_int;
      Int64.min_int;
    ]

let law_bits (w, w') =
  List.iter
    (fun (fmt, f) ->
      cover ("a tie of " ^ fmt)
        (is_tie f (Int64.float_of_bits w) || is_tie f (Int64.float_of_bits w')))
    rounded_formats;
  List.iter
    (fun (D.Any dt) ->
      let x = value_of_words dt w w' in
      equal ~msg:(D.name dt) string (hex (array_bits dt x)) (hex (P.bits dt x)))
    D.all

(* A NaN's bits are those a store of it gives (Dtype's stores): quiet, of its
   sign, with the top of its payload. Each row is a double's bits, then the
   float64 and float32 bits of the store. *)
let nan_stores =
  [
    (0xFFF4000000000000L, 0xFFFC000000000000L, 0xFFE00000l);
    (0x7FF0040000000001L, 0x7FF8040000000001L, 0x7FC02000l);
    (0x7FF8000000000000L, 0x7FF8000000000000L, 0x7FC00000l);
    (0x7FFFFFFFFFFFFFFFL, 0x7FFFFFFFFFFFFFFFL, 0x7FFFFFFFl);
  ]

let test_nan_bits (x, b64, b32) =
  let x = Int64.float_of_bits x in
  let word64 s at = String.get_int64_le s at
  and word32 s at = String.get_int32_le s at in
  equal ~msg:"float64" int64 b64 (word64 (P.bits D.Float64 x) 0);
  equal ~msg:"float32" int32 b32 (word32 (P.bits D.Float32 x) 0);
  let c = { Complex.re = x; im = x } in
  let c128 = P.bits D.Complex128 c and c64 = P.bits D.Complex64 c in
  equal ~msg:"complex128" (pair int64 int64) (b64, b64)
    (word64 c128 0, word64 c128 8);
  equal ~msg:"complex64" (pair int32 int32) (b32, b32)
    (word32 c64 0, word32 c64 4)

(* The integer dtypes OCaml writes as [int], with the range each name states. *)
type int_dtype = Ints : string * (int, 's) D.t * int * int -> int_dtype

let int_dtypes =
  [
    Ints ("int16", D.Int16, -32768, 32767);
    Ints ("uint16", D.Uint16, 0, 65535);
    Ints ("int8", D.Int8, -128, 127);
    Ints ("uint8", D.Uint8, 0, 255);
    Ints ("int4", D.Int4, -8, 7);
    Ints ("uint4", D.Uint4, 0, 15);
  ]

let test_int_range (Ints (_, dt, lo, hi)) =
  raises_match ~msg:"below" Exn.invalid_arg (fun () -> P.bits dt (lo - 1));
  raises_match ~msg:"above" Exn.invalid_arg (fun () -> P.bits dt (hi + 1));
  equal ~msg:"least" string (hex (array_bits dt lo)) (hex (P.bits dt lo));
  equal ~msg:"greatest" string (hex (array_bits dt hi)) (hex (P.bits dt hi))

(* v *)

let law_v d =
  let has f = Array.exists f d.nodes in
  cover "an Op1" (has (function P.Op1 _ -> true | _ -> false));
  cover "an Op2" (has (function P.Op2 _ -> true | _ -> false));
  cover "an Op3" (has (function P.Op3 _ -> true | _ -> false));
  let p = program d in
  equal ~msg:"ins" (list string)
    (Array.to_list (Array.map name d.ins))
    (List.map name (Iarray.to_list p.ins));
  equal ~msg:"nodes" (array node) d.nodes (Iarray.to_array p.nodes);
  equal ~msg:"outs" (array int) d.outs (Iarray.to_array p.outs)

(* Whether [of_node], v over the operands' nodes, builds [n] over [ins]. *)
let builds ins n =
  match P.of_node ~ins:(Iarray.of_array ins) n with
  | _ -> true
  | exception Invalid_argument _ -> false

let test_v_kinds = function
  | K1 k ->
      List.iter
        (fun (D.Any x as ax) ->
          equal ~msg:(name ax) (list string)
            (names_where (fun (D.Any y) -> P.accepts1 k x y))
            (names_where (fun y -> builds [| ax |] (Op1 (k, y, 0)))))
        D.all
  | K2 k ->
      equal (list string)
        (names_where (fun (D.Any x) -> P.accepts2 k x))
        (names_where (fun x -> builds [| x; x |] (Op2 (k, 0, 1))))
  | K3 k ->
      List.iter
        (fun (D.Any c as ac) ->
          equal ~msg:(name ac) (list string)
            (names_where (fun (D.Any x) -> P.accepts3 k c x))
            (names_where (fun x -> builds [| ac; x; x |] (Op3 (k, 0, 1, 2)))))
        D.all

(* Each row: a program v refuses, as its operands, nodes and outputs. *)
let ill_formed : (string * D.any array * P.node array * int array) list =
  [
    ("a forward reference", [| f32 |], [| Op1 (Copy, f32, 1); In 0 |], [| 0 |]);
    ("a reference to itself", [| f32 |], [| Op1 (Copy, f32, 0) |], [| 0 |]);
    ("a negative reference", [| f32 |], [| In 0; Op1 (Copy, f32, -1) |], [| 1 |]);
    ("an operand past ins", [| f32 |], [| In 1 |], [| 0 |]);
    ("a negative operand", [| f32 |], [| In (-1) |], [| 0 |]);
    ( "an axis past max_rank",
      [||],
      [| Coord Nx_array.Layout.max_rank |],
      [| 0 |] );
    ("a negative axis", [||], [| Coord (-1) |], [| 0 |]);
    ( "a kind off its domain",
      [| f32 |],
      [| In 0; Op2 (Binary Idiv, 0, 0) |],
      [| 1 |] );
    ( "two dtypes",
      [| f32; f64 |],
      [| In 0; In 1; Op2 (Binary Add, 0, 1) |],
      [| 2 |] );
    ( "a where over branches of two dtypes",
      [| boolean; f32; f64 |],
      [| In 0; In 1; In 2; Op3 (Where, 0, 1, 2) |],
      [| 3 |] );
    ( "an fma whose last two operands differ in dtype",
      [| f32; f64 |],
      [| In 0; In 1; Op3 (Fma, 0, 0, 1) |],
      [| 2 |] );
    ( "an fma whose first operand is of another dtype",
      [| f32; f64 |],
      [| In 0; In 1; Op3 (Fma, 1, 0, 0) |],
      [| 2 |] );
    ("bits of the wrong width", [||], [| Const (f32, "\000\000") |], [| 0 |]);
    ("a bool of no byte", [||], [| Const (boolean, "") |], [| 0 |]);
    ("a bool of two bytes", [||], [| Const (boolean, "\001\001") |], [| 0 |]);
    ("a bit of 2", [||], [| Const (bit, "\002") |], [| 0 |]);
    ("an int4 of 16", [||], [| Const (D.Any D.Int4, "\016") |], [| 0 |]);
    ("no output", [| f32 |], [| In 0 |], [||]);
    ("an output past the nodes", [| f32 |], [| In 0 |], [| 1 |]);
    ("a negative output", [| f32 |], [| In 0 |], [| -1 |]);
  ]

let test_ill_formed (_, ins, nodes, outs) =
  raises_match Exn.invalid_arg (fun () -> program { ins; nodes; outs })

(* A Const of [dt] and bits [b] as Where's condition between two float32
   operands, 1. and 2.: the first where it is true. *)
let bool_consts =
  [
    ("a bool of 0", boolean, "\000", false);
    ("a bool of 1", boolean, "\001", true);
    ("a bool of 2", boolean, "\002", true);
    ("a bool of 128", boolean, "\128", true);
    ("a bool of 255", boolean, "\255", true);
    ("a bit of 0", bit, "\000", false);
    ("a bit of 1", bit, "\001", true);
  ]

let test_bool_const (_, dt, b, holds) =
  let p =
    program
      {
        ins = [| f32; f32 |];
        nodes = [| Const (dt, b); In 0; In 1; Op3 (Where, 0, 1, 2) |];
        outs = [| 3 |];
      }
  in
  let one x = operand1 f32 (P.bits D.Float32 x) in
  let want = P.bits D.Float32 (if holds then 1. else 2.) in
  equal outputs (Ok [| [| want |] |]) (compute p [| one 1.; one 2. |])

(* Twenty operands, each an output; an expression of operand 20; a node over
   operand 19. *)
let test_many_operands () =
  let ins n = Iarray.init n (fun _ -> f32) in
  let p =
    P.v ~ins:(ins 20)
      (Iarray.init 20 (fun k -> P.In k))
      ~outs:(Iarray.init 20 Fun.id)
  in
  equal ~msg:"v" (array int) (Array.init 20 Fun.id) (Iarray.to_array p.outs);
  let q = P.of_exprs ~ins:(ins 21) [| P.E.(any (input 20 D.Float32)) |] in
  equal ~msg:"of_exprs" (array node) [| P.In 20 |] (Iarray.to_array q.nodes);
  let r = P.of_node ~ins:(ins 20) (Op2 (Binary Add, 0, 19)) in
  equal ~msg:"of_node" node (Op2 (Binary Add, 0, 19)) (Iarray.get r.nodes 20)

(* Its 8 words, the words it allocated when the budget was set, are the program
   and its nodes' dtypes; flambda inlines some allocations away. *)
let test_v_allocates () =
  let ins : D.any iarray = [| f32 |] and outs : int iarray = [| 2 |] in
  let nodes : P.node iarray =
    [| P.In 0; P.Const (f32, P.bits D.Float32 2.); P.Op2 (P.Binary Mul, 0, 1) |]
  in
  at_most (float 0.5) ~than:8. (words (fun () -> P.v ~ins nodes ~outs))

let law_holds_as_v (d, off) =
  let check what (p : P.t) =
    equal ~msg:what structural p (P.v ~ins:p.ins p.nodes ~outs:p.outs)
  in
  let ins = Iarray.of_array d.ins in
  let exprs = Array.map any (typed d) in
  check "of_exprs"
    (P.of_exprs ~ins (Iarray.of_array (Array.map (Array.get exprs) d.outs)));
  check "subst" (subst ~ins:d.ins (shift off) d)

(* of_node *)

(* One node over drawn operands, its references in and around the operands'
   range. *)
let one_node =
  let open Gen in
  with_pp
    (fun ppf (ins, n) ->
      Format.fprintf ppf "%a over [%s]" pp_node n
        (String.concat "; " (Array.to_list (Array.map name ins))))
    (let* ins =
       array ~size:(int_range 0 3) (of_list ~pp:pp_dtype base_dtypes)
     in
     let k = Array.length ins in
     let ref_ = int_range (-1) (k + 1) in
     let+ n =
       one_of
         [
           map (fun j -> P.In j) ref_;
           map (fun a -> P.Coord a) (int_range 0 Nx_array.Layout.max_rank);
           constant ~pp:pp_node (P.Const (f32, P.bits D.Float32 1.5));
           map
             (fun j ->
               P.Op1 (Copy, (if j >= 0 && j < k then ins.(j) else f32), j))
             ref_;
           map (fun (j, l) -> P.Op2 (Binary Add, j, l)) (pair ref_ ref_);
           map (fun (c, j, l) -> P.Op3 (Where, c, j, l)) (triple ref_ ref_ ref_);
         ]
     in
     (ins, n))

let law_of_node (ins, n) =
  let k = Array.length ins in
  let built f =
    match f () with p -> Some p | exception Invalid_argument _ -> None
  in
  let by_v =
    built (fun () ->
        P.v ~ins:(Iarray.of_array ins)
          (Iarray.init (k + 1) (fun i -> if i < k then P.In i else n))
          ~outs:[| k |])
  in
  cover "built" (Option.is_some by_v);
  cover "refused" (Option.is_none by_v);
  equal (option structural) by_v
    (built (fun () -> P.of_node ~ins:(Iarray.of_array ins) n))

(* dtype *)

(* Each node's dtype as dtype states it: its operand's for In, int64 for Coord,
   the one it names for Const and Op1, its operands' for a Binary kind, bool for
   a Compare kind, and its last two operands' for Op3. *)
let stated_dtypes d =
  let t = Array.make (Array.length d.nodes) i64 in
  Array.iteri
    (fun i n ->
      t.(i) <-
        (match n with
        | P.In k -> d.ins.(k)
        | Coord _ -> i64
        | Const (dt, _) | Op1 (_, dt, _) -> dt
        | Op2 (Binary _, a, _) -> t.(a)
        | Op2 (Compare _, _, _) -> boolean
        | Op3 (_, _, b, _) -> t.(b)))
    d.nodes;
  t

let law_dtype d =
  let p = program d in
  equal (array string)
    (Array.map name (stated_dtypes d))
    (Array.init (Array.length d.nodes) (fun i -> name (P.dtype p i)))

let test_dtype_refuses at =
  let p = P.v ~ins:[| f32 |] [| P.In 0 |] ~outs:[| 0 |] in
  raises_match Exn.invalid_arg (fun () -> P.dtype p at)

(* is_identity *)

let identity = { ins = [| f32 |]; nodes = [| P.In 0 |]; outs = [| 0 |] }

(* Programs beside the identity that is_identity tells apart. *)
let near_identities =
  [
    identity;
    { identity with nodes = [| In 0; Op1 (Copy, f32, 0) |]; outs = [| 1 |] };
    { identity with nodes = [| In 0; In 0 |]; outs = [| 1 |] };
    { identity with outs = [| 0; 0 |] };
    { identity with ins = [| f32; f32 |] };
    { identity with ins = [||]; nodes = [| Const (f32, P.bits D.Float32 1.) |] };
  ]

let law_identity d =
  let want =
    Array.length d.ins = 1
    && Array.length d.outs = 1
    && d.nodes.(d.outs.(0)) = P.In 0
  in
  cover "the identity" want;
  cover "not the identity" (not want);
  satisfies
    ~claim:(if want then "is_identity" else "not is_identity")
    structural
    (fun p -> P.is_identity p = want)
    (program d)

(* single and is_single *)

(* Whether [n] reads [k] operands, each once, in order. *)
let reads_in_order k (n : P.node) =
  match n with
  | Const _ | Coord _ -> k = 0
  | Op1 (_, _, 0) -> k = 1
  | Op2 (_, 0, 1) -> k = 2
  | Op3 (_, 0, 1, 2) -> k = 3
  | In _ | Op1 _ | Op2 _ | Op3 _ -> false

(* A kind of two and of three operands reading them in order, and one out of
   order, which one_node seldom draws. *)
let singles =
  [
    ([| f32; f32 |], P.Op2 (Compare Less, 0, 1));
    ([| boolean; f32; f32 |], P.Op3 (Where, 0, 1, 2));
    ([| f32; f32 |], P.Op2 (Binary Sub, 1, 0));
  ]

let law_single (ins, n) =
  let p =
    match P.of_node ~ins:(Iarray.of_array ins) n with
    | p -> Some p
    | exception Invalid_argument _ -> None
  in
  let in_order = reads_in_order (Array.length ins) n in
  cover "a node reading its operands in order" (Option.is_some p && in_order);
  cover "a node reading them otherwise" (Option.is_some p && not in_order);
  cover "three operands in order"
    (Option.is_some p && in_order && Array.length ins = 3);
  match p with
  | None -> ()
  | Some p ->
      equal (option node) (if in_order then Some n else None) (P.single p)

(* Programs of two outputs and of two kind nodes. *)
let not_of_node =
  [
    ( "two outputs",
      {
        ins = [| f32 |];
        nodes = [| In 0; Op1 (Unary Neg, f32, 0) |];
        outs = [| 1; 1 |];
      } );
    ( "two kind nodes",
      {
        ins = [| f32 |];
        nodes = [| In 0; Op1 (Unary Neg, f32, 0); Op1 (Unary Neg, f32, 1) |];
        outs = [| 2 |];
      } );
  ]

let test_not_single (_, d) = equal (option node) None (P.single (program d))

let law_is_single d =
  let p = program d in
  let last = Iarray.get p.nodes (Iarray.length p.nodes - 1) in
  let want = P.single p = Some last in
  cover "a single program" want;
  cover "another program" (not want);
  satisfies
    ~claim:(if want then "is_single" else "not is_single")
    structural
    (fun p -> P.is_single p = want)
    p

let is_single_examples =
  List.map
    (fun (ins, n) -> drawn_of (P.of_node ~ins:(Iarray.of_array ins) n))
    singles
  @ List.map snd not_of_node

let test_is_single_allocates () =
  let single = P.of_node ~ins:[| f32; f32 |] (Op2 (Binary Add, 0, 1)) in
  let two = program (List.assoc "two kind nodes" not_of_node) in
  at_most ~msg:"a single program" (float 0.5) ~than:0.
    (words (fun () -> P.is_single single));
  at_most ~msg:"two kind nodes" (float 0.5) ~than:0.
    (words (fun () -> P.is_single two))

(* E *)

let test_expr_dtypes () =
  let x = P.E.input 0 D.Float32 in
  equal (list string)
    [ "float16"; "int64"; "int8"; "bool"; "uint4" ]
    P.E.
      [
        D.name (dtype (input 1 D.Float16));
        D.name (dtype (coord 0));
        D.name (dtype (cast D.Int8 x));
        D.name (dtype (less x x));
        D.name (dtype (const D.Uint4 3));
      ]

(* Operands of the kind functions' rows: two float32, a bool, a uint64, a
   complex64 and an int32. *)
let fn_ins = [| f32; f32; boolean; D.Any D.Uint64; D.Any D.Complex64; i32 |]

(* Each row: a kind function's expression, then its kind's constructor's. *)
let kind_functions =
  let x = P.E.input 0 D.Float32 and y = P.E.input 1 D.Float32 in
  let c = P.E.input 2 D.Bool and w = P.E.input 3 D.Uint64 in
  let z = P.E.input 4 D.Complex64 and n = P.E.input 5 D.Int32 in
  let u k f =
    (unary_name k, P.E.any (f x), P.E.(any (op1 (Unary k) D.Float32 x)))
  in
  let uz k f =
    (unary_name k, P.E.any (f z), P.E.(any (op1 (Unary k) D.Complex64 z)))
  in
  let b k f = (binary_name k, P.E.any (f x y), P.E.(any (binary k x y))) in
  let bi k f =
    (binary_name k ^ " on int32", P.E.any (f n n), P.E.(any (binary k n n)))
  in
  let bw k f =
    (binary_name k ^ " on uint64", P.E.any (f w w), P.E.(any (binary k w w)))
  in
  let cmp k f = (compare_name k, P.E.any (f x y), P.E.(any (compare k x y))) in
  P.E.
    [
      u Neg neg;
      u Recip recip;
      u Abs abs;
      u Sign sign;
      u Sqrt sqrt;
      u Exp exp;
      u Exp2 exp2;
      u Log log;
      u Log2 log2;
      u Log1p log1p;
      u Expm1 expm1;
      u Sin sin;
      u Cos cos;
      u Tan tan;
      u Asin asin;
      u Acos acos;
      u Atan atan;
      u Sinh sinh;
      u Cosh cosh;
      u Tanh tanh;
      uz Asinh asinh;
      uz Acosh acosh;
      uz Atanh atanh;
      u Erf erf;
      u Floor floor;
      u Ceil ceil;
      u Round round;
      u Trunc trunc;
      b Add add;
      b Sub sub;
      b Mul mul;
      b Fdiv fdiv;
      b Mod rem;
      b Pow pow;
      b Atan2 atan2;
      b Maximum maximum;
      b Minimum minimum;
      bi Idiv idiv;
      bi And logand;
      bi Or logor;
      bw Xor logxor;
      bw Threefry threefry;
      cmp Equal equal;
      cmp Not_equal not_equal;
      cmp Less less;
      cmp Less_equal less_equal;
      ("copy", any (copy x), any (op1 Copy D.Float32 x));
      ("cast", any (cast D.Int8 x), any (op1 Cast D.Int8 x));
      ("bitcast", any (bitcast D.Int32 x), any (op1 Bitcast D.Int32 x));
      ("where", any (where c x y), any (op3 Where c x y));
      ("fma", any (fma x y x), any (op3 Fma x y x));
    ]

let of_expr ~ins e = P.of_exprs ~ins:(Iarray.of_array ins) [| e |]

let test_kind_function (_, got, want) =
  equal structural (of_expr ~ins:fn_ins want) (of_expr ~ins:fn_ins got)

let test_cast_own_dtype () =
  let x = P.E.input 0 D.Float32 in
  equal (array node) [| P.In 0 |]
    (Iarray.to_array (of_expr ~ins:[| f32 |] P.E.(any (cast D.Float32 x))).nodes)

(* Each operator over two operands lowers as the function prog.mli names, [>]
   and [>=] with their operands swapped. *)
let operators =
  let x = P.E.input 0 D.Float32 and y = P.E.input 1 D.Float32 in
  let c = P.E.input 0 D.Bool and e = P.E.input 1 D.Bool in
  let v = P.E.input 0 D.Uint64 and w = P.E.input 1 D.Uint64 in
  let u64 = D.Any D.Uint64 in
  P.E.
    [
      ("~-", f32, any ~-x, any (neg x));
      ("+", f32, any (x + y), any (add x y));
      ("-", f32, any (x - y), any (sub x y));
      ("*", f32, any (x * y), any (mul x y));
      ("/ on floats", f32, any (x / y), any (fdiv x y));
      ("/ on integers", u64, any (v / w), any (idiv v w));
      ("mod", f32, any (x mod y), any (rem x y));
      ("**", f32, any (x ** y), any (pow x y));
      ("land", u64, any (v land w), any (logand v w));
      ("lor", u64, any (v lor w), any (logor v w));
      ("lxor", u64, any (v lxor w), any (logxor v w));
      ("&&", boolean, any (c && e), any (logand c e));
      ("||", boolean, any (c || e), any (logor c e));
      ("=", f32, any (x = y), any (equal x y));
      ("<>", f32, any (x <> y), any (not_equal x y));
      ("<", f32, any (x < y), any (less x y));
      ("<=", f32, any (x <= y), any (less_equal x y));
      (">", f32, any (x > y), any (less y x));
      (">=", f32, any (x >= y), any (less_equal y x));
    ]

let test_operator (_, dt, e, want) =
  equal structural
    (of_expr ~ins:[| dt; dt |] want)
    (of_expr ~ins:[| dt; dt |] e)

(* [f ()] returns, or the test fails naming what it refused. *)
let takes ~msg f =
  match f () with
  | _ -> ()
  | exception Invalid_argument m -> failf "%s: refused (%s)" msg m

let refuses ?msg f = raises_match ?msg Exn.invalid_arg f

let test_constructors () =
  List.iter
    (fun (D.Any a) ->
      let x = P.E.input 0 a in
      List.iter
        (fun (D.Any y) ->
          List.iter
            (fun k ->
              let msg =
                strf "op1 %s %s %s" (op1_name k) (D.name a) (D.name y)
              in
              let f () = P.E.op1 k y x in
              if P.accepts1 k a y then takes ~msg f else refuses ~msg f)
            op1s)
        D.all;
      List.iter
        (fun (k, kname) ->
          let msg = strf "binary %s %s" kname (D.name a) in
          let f () = P.E.binary k x x in
          if P.accepts2 (Binary k) a then takes ~msg f else refuses ~msg f)
        binaries;
      List.iter
        (fun (D.Any c) ->
          let c = P.E.input 1 c in
          List.iter
            (fun k ->
              let msg =
                strf "op3 %s %s %s" (op3_name k)
                  (D.name (P.E.dtype c))
                  (D.name a)
              in
              let f () = P.E.op3 k c x x in
              if P.accepts3 k (P.E.dtype c) a then takes ~msg f
              else refuses ~msg f)
            op3s)
        D.all)
    D.all

(* The kind functions whose operands any dtype types: each checks its operands
   as v does. *)
type unary_fn = { unary : P.unary; f : 'v 's. ('v, 's) P.E.t -> ('v, 's) P.E.t }

type binary_fn = {
  binary : P.binary;
  g : 'v 's. ('v, 's) P.E.t -> ('v, 's) P.E.t -> ('v, 's) P.E.t;
}

let unary_fns =
  P.E.
    [
      { unary = Neg; f = neg };
      { unary = Recip; f = recip };
      { unary = Abs; f = abs };
      { unary = Sign; f = sign };
      { unary = Sqrt; f = sqrt };
      { unary = Exp; f = exp };
      { unary = Exp2; f = exp2 };
      { unary = Log; f = log };
      { unary = Log2; f = log2 };
      { unary = Sin; f = sin };
      { unary = Cos; f = cos };
      { unary = Tan; f = tan };
      { unary = Asin; f = asin };
      { unary = Acos; f = acos };
      { unary = Atan; f = atan };
      { unary = Sinh; f = sinh };
      { unary = Cosh; f = cosh };
      { unary = Tanh; f = tanh };
      { unary = Asinh; f = asinh };
      { unary = Acosh; f = acosh };
      { unary = Atanh; f = atanh };
      { unary = Floor; f = floor };
      { unary = Ceil; f = ceil };
      { unary = Round; f = round };
      { unary = Trunc; f = trunc };
    ]

let binary_fns =
  P.E.
    [
      { binary = Add; g = add };
      { binary = Sub; g = sub };
      { binary = Mul; g = mul };
      { binary = Fdiv; g = fdiv };
      { binary = Idiv; g = idiv };
      { binary = Mod; g = rem };
      { binary = Pow; g = pow };
      { binary = Maximum; g = maximum };
      { binary = Minimum; g = minimum };
      { binary = And; g = logand };
      { binary = Or; g = logor };
      { binary = Xor; g = logxor };
    ]

let test_unary_fn r =
  List.iter
    (fun (D.Any dt) ->
      let x = P.E.input 0 dt in
      let msg = D.name dt and f () = r.f x in
      if P.accepts1 (Unary r.unary) dt dt then takes ~msg f else refuses ~msg f)
    D.all

let test_binary_fn r =
  List.iter
    (fun (D.Any dt) ->
      let x = P.E.input 0 dt in
      let msg = D.name dt and f () = r.g x x in
      if P.accepts2 (Binary r.binary) dt then takes ~msg f else refuses ~msg f)
    D.all

let leaf_refusals : (string * (unit -> unit)) list =
  [
    ("input -1", fun () -> ignore (P.E.input (-1) D.Float32));
    ("coord -1", fun () -> ignore (P.E.coord (-1)));
    ("coord max_rank", fun () -> ignore (P.E.coord Nx_array.Layout.max_rank));
    ("const int8 128", fun () -> ignore (P.E.const D.Int8 128));
    ("const uint4 -1", fun () -> ignore (P.E.const D.Uint4 (-1)));
  ]

let test_leaf_refuses (_, f) = refuses f

(* of_exprs *)

(* A -0. of float32 and an int32 of the same bits, both outputs, which drawn
   programs reach one run in a few. *)
let signed_zero =
  {
    ins = [||];
    nodes =
      [|
        Const (f32, P.bits D.Float32 (-0.));
        Const (i32, P.bits D.Int32 Int32.min_int);
      |];
    outs = [| 0; 1 |];
  }

(* Whether two of [q]'s constants have the same bits and two dtypes. *)
let shared_constant (q : P.t) =
  let consts =
    List.filter_map
      (function P.Const (d, b) -> Some (d, b) | _ -> None)
      (Iarray.to_list q.nodes)
  in
  List.exists
    (fun (d, b) ->
      List.exists (fun (d', b') -> b = b' && not (same d d')) consts)
    consts

let law_of_exprs d =
  (* The odd outputs from a second build, so that every node they reach is a
     value apart from the first build's and merges by its structure. *)
  let first = Array.map any (typed d) and second = Array.map any (typed d) in
  let out k i = if k mod 2 = 0 then first.(i) else second.(i) in
  let q =
    P.of_exprs ~ins:(Iarray.of_array d.ins)
      (Iarray.of_array (Array.mapi out d.outs))
  in
  let n = Iarray.length q.nodes in
  let reached =
    Array.fold_left (fun a r -> if r then a + 1 else a) 0 (reach d d.outs)
  in
  let has f = Iarray.exists f q.nodes in
  let float_const (f : float -> bool) =
    has (function
      | P.Const (D.Any dt, b) -> (
          match D.kind dt with Float -> f (element dt b) | _ -> false)
      | _ -> false)
  in
  cover "at most 40 nodes" (n <= 40);
  cover "41 to 80 nodes" (n > 40 && n <= 80);
  cover "over 128 nodes" (n > 128);
  cover "outputs of two builds" (Array.length d.outs >= 2);
  cover "a merge" (n < reached);
  cover "a node unreached" (reached < Array.length d.nodes);
  let rec repeats = function
    | [] -> false
    | o :: os -> List.mem o os || repeats os
  in
  cover "an output repeated" (repeats (Array.to_list d.outs));
  cover "a -0. constant" (float_const (fun x -> x = 0. && Float.sign_bit x));
  cover "a NaN constant" (float_const Float.is_nan);
  cover "one constant's bits in two dtypes" (shared_constant q);
  cover "a Where over two nodes"
    (has (function P.Op3 (Where, _, b, c) -> b <> c | _ -> false));
  cover "a Where over one node"
    (has (function P.Op3 (Where, _, b, c) -> b = c | _ -> false));
  cover "an Fma over three nodes"
    (has (function
      | P.Op3 (Fma, a, b, c) -> a <> b && b <> c && a <> c
      | _ -> false));
  cover "an Fma over one node"
    (has (function P.Op3 (Fma, a, b, c) -> a = b && b = c | _ -> false));
  let nodes, outs = lowered d in
  equal ~msg:"nodes" (array node) nodes (Iarray.to_array q.nodes);
  equal ~msg:"outs" (array int) outs (Iarray.to_array q.outs);
  equal ~msg:"ins" (list string)
    (Array.to_list (Array.map name d.ins))
    (List.map name (Iarray.to_list q.ins))

let test_of_exprs_two_builds () =
  (* Two builds of one chain of 15000 nodes, each node a constant or a sum over
     the last: apart as values, one program, both outputs its last node. *)
  let chain () =
    let y = ref (P.E.input 0 D.Int64) in
    for i = 1 to 7499 do
      y := P.E.(!y + const D.Int64 (Int64.of_int i))
    done;
    P.E.any !y
  in
  let q = P.of_exprs ~ins:[| i64 |] [| chain (); chain () |] in
  equal ~msg:"nodes" int 14999 (Iarray.length q.nodes);
  equal ~msg:"outs" (array int) [| 14998; 14998 |] (Iarray.to_array q.outs)

let test_of_exprs_fresh_copies () =
  (* A sum of two fresh builds of one sum, 16 levels deep: 2^17 - 1 expressions,
     apart as values, of 17 nodes. A lowering that scanned its walked
     expressions while it held few nodes would compare billions of pairs. *)
  let rec tree d =
    if d = 0 then P.E.input 0 D.Int64
    else
      let a = tree (d - 1) in
      let b = tree (d - 1) in
      P.E.(a + b)
  in
  let q = P.of_exprs ~ins:[| i64 |] [| P.E.any (tree 16) |] in
  let want =
    Array.init 17 (fun i : P.node ->
        if i = 0 then In 0 else Op2 (Binary Add, i - 1, i - 1))
  in
  equal ~msg:"nodes" (array node) want (Iarray.to_array q.nodes);
  equal ~msg:"outs" (array int) [| 16 |] (Iarray.to_array q.outs)

let of_exprs_refusals =
  let x = P.E.any (P.E.input 0 D.Float32) in
  [
    ("no output", [| f32 |], [||]);
    ("an operand past ins", [||], [| x |]);
    ("an operand of another dtype", [| i32 |], [| x |]);
  ]

let test_of_exprs_refuses (_, ins, outs) =
  refuses (fun () ->
      P.of_exprs ~ins:(Iarray.of_array ins) (Iarray.of_array outs))

(* subst *)

(* of_exprs's program over [ins] of [d]'s outputs [picked], each leaf replaced
   as [f] says: subst's result as prog.mli defines it. *)
let substituted ~ins f d picked =
  let e = typed ~leaf:f d in
  P.of_exprs ~ins:(Iarray.of_array ins)
    (Iarray.of_array (Array.map (fun o -> any e.(o)) picked))

let law_subst (d, off) =
  let reached = reach d d.outs in
  let coord f =
    List.exists Fun.id
      (List.mapi
         (fun i nd ->
           match nd with P.Coord c -> reached.(i) && f off.(c) | _ -> false)
         (Array.to_list d.nodes))
  in
  cover "a coordinate shifted" (coord (fun o -> o <> 0));
  cover "a coordinate kept" (coord (fun o -> o = 0));
  cover "no leaf replaced" (Array.for_all (fun o -> o = 0) off);
  let s = subst ~ins:d.ins (shift off) d in
  cover "fewer nodes than drawn" (Iarray.length s.nodes < Array.length d.nodes);
  equal structural (substituted ~ins:d.ins (shift off) d d.outs) s

(* A drawn program, and one to four outputs picked among its own by index. *)
let selecting =
  let open Gen in
  with_pp
    (fun ppf (d, sel) ->
      Format.fprintf ppf "%a, picking %a" pp_drawn d Elements.pp_ints sel)
    (let+ d = drawn
     and+ sel = array ~size:(int_range 1 4) (int_range 0 1_000) in
     (d, Array.map (fun k -> k mod Array.length d.outs) sel))

let keep _ = None

let law_subst_calls (d, sel) =
  let calls = ref [] in
  let f n =
    calls := n :: !calls;
    None
  in
  ignore
    (P.subst ~outs:(Iarray.of_array sel) ~ins:(Iarray.of_array d.ins) f
       (program d));
  let picked = Array.map (Array.get d.outs) sel in
  let reached = reach d picked in
  let is_leaf = function P.In _ | Coord _ | Const _ -> true | _ -> false in
  let leaves =
    List.filteri (fun i nd -> reached.(i) && is_leaf nd) (Array.to_list d.nodes)
  in
  cover "a leaf unreached"
    (Array.exists Fun.id
       (Array.mapi (fun i nd -> is_leaf nd && not reached.(i)) d.nodes));
  equal (list node) (List.sort compare leaves) (List.sort compare !calls)

let law_subst_outs (d, sel) =
  let n = Array.length sel in
  cover "an output twice"
    (List.length (List.sort_uniq Int.compare (Array.to_list sel)) < n);
  cover "an output dropped"
    (List.exists
       (fun o -> not (Array.mem o sel))
       (List.init (Array.length d.outs) Fun.id));
  cover "outputs reordered"
    (List.exists (fun j -> sel.(j) > sel.(j + 1)) (List.init (n - 1) Fun.id));
  equal structural
    (substituted ~ins:d.ins keep d (Array.map (Array.get d.outs) sel))
    (subst ~outs:(Iarray.of_array sel) ~ins:d.ins keep d)

let without k a =
  Array.of_list (List.filteri (fun i _ -> i <> k) (Array.to_list a))

(* The substitution that binds operand [k] of [ins] to [x]: [In k] is [x], and
   [In i] for [i > k] is [In (i - 1)]. *)
let binds ins k x : P.node -> ex option = function
  | P.In i when i = k -> Some x
  | In i when i > k ->
      let (D.Any dt) = ins.(i) in
      Some (Ex (P.E.input (i - 1) dt))
  | In _ | Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _ -> None

(* [d]'s operand [k] bound to [x], an expression over its other operands: the
   program over them that substituted states. *)
let check_bind d k x =
  let others = without k d.ins in
  let s = subst ~ins:others (binds d.ins k x) d in
  equal ~msg:"ins" (list string)
    (Array.to_list (Array.map name others))
    (List.map name (Iarray.to_list s.ins));
  equal structural (substituted ~ins:others (binds d.ins k x) d d.outs) s

(* A drawn program over one operand or more, an operand [k] of it, and an
   expression drawn over the others, cast to operand [k]'s dtype. *)
let binding =
  let open Gen in
  with_pp
    (fun ppf (d, k, x) ->
      Format.fprintf ppf "%a,@ operand %d,@ bound to %a" pp_drawn d k pp_drawn x)
    (let* d = drawn_over (int_range 1 3) in
     let+ k = int_range 0 (Array.length d.ins - 1)
     and+ n = int_range 1 6
     and+ choices = array ~size:(constant 24) (int_range 0 0x3FFF_FFFF) in
     (d, k, grow (without k d.ins) choices n 1))

let law_bind (d, k, x) =
  cover "p reads operand k" (Array.mem (P.In k) d.nodes);
  cover "an operand after k" (k < Array.length d.ins - 1);
  cover "k is 0" (k = 0);
  let (D.Any dt) = d.ins.(k) in
  let (Ex e) = (typed x).(x.outs.(0)) in
  check_bind d k (Ex (P.E.cast dt e))

let test_bind_leaf (_, d, k, x) = check_bind (drawn_of d) k x

let bind_leaves =
  let p ins nodes outs =
    P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
      ~outs:(Iarray.of_array outs)
  in
  [
    ( "an operand",
      p [| f32; f32 |] [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |] [| 2 |],
      0,
      Ex (P.E.input 0 D.Float32) );
    ( "a coordinate",
      p [| i64; f32 |]
        [| P.In 0; In 1; Op1 (Cast, f32, 0); Op2 (Binary Mul, 2, 1) |]
        [| 3 |],
      0,
      Ex (P.E.coord 1) );
    ( "a constant, output as p's In k",
      p [| f32; f64 |] [| P.In 0; In 1 |] [| 0; 1 |],
      0,
      Ex (P.E.const D.Float32 2.5) );
  ]

(* A program of a float32 operand, a float64 operand and a coordinate, each an
   output. *)
let three_leaves () =
  P.of_exprs ~ins:[| f32; f64 |]
    P.E.[| any (input 0 D.Float32); any (input 1 D.Float64); any (coord 0) |]

let in1 x : P.node -> P.E.any option = function P.In 1 -> Some x | _ -> None

let test_replaced_leaf () =
  let s =
    P.subst ~ins:[| f32 |]
      (in1 P.E.(any (const D.Float64 1.)))
      (three_leaves ())
  in
  equal (list string) [ "float32" ] (List.map name (Iarray.to_list s.ins))

let test_unreached_leaf () =
  let unreached : P.node -> P.E.any option = function
    | P.In 1 -> failwith "subst passed f a leaf no picked output reaches"
    | _ -> None
  in
  let s = P.subst ~outs:[| 0; 2 |] ~ins:[| f32 |] unreached (three_leaves ()) in
  equal (array node) [| P.In 0; Coord 0 |] (Iarray.to_array s.nodes)

let subst_refusals =
  [
    ( "an expression of another dtype",
      None,
      [| f32 |],
      in1 P.E.(any (const D.Float32 1.)) );
    ( "a coordinate's expression of another dtype",
      None,
      [| f32; f64 |],
      function P.Coord _ -> Some P.E.(any (const D.Int32 1l)) | _ -> None );
    ( "a read of an operand the result lacks",
      None,
      [| f32 |],
      in1 P.E.(any (cast D.Float64 (input 1 D.Float32))) );
    ( "a read of an operand at another dtype",
      None,
      [| f32 |],
      in1 P.E.(any (input 0 D.Float64)) );
    ("a kept operand past ins", None, [| f32 |], keep);
    ("a kept operand at another dtype", None, [| f32; f32 |], keep);
    ("an output past p's", Some [| 3 |], [| f32; f64 |], keep);
    ("a negative output", Some [| -1 |], [| f32; f64 |], keep);
    ("no output", Some [||], [| f32; f64 |], keep);
  ]

let test_subst_refuses (_, outs, ins, f) =
  let outs = Option.map Iarray.of_array outs in
  refuses (fun () ->
      P.subst ?outs ~ins:(Iarray.of_array ins) f (three_leaves ()))

(* Ranges *)

(* An integer or boolean element as its value in an int64, unsigned ones
   zero-extended and compared unsigned, booleans 0 or 1. *)
let unsigned (D.Any dt) = D.is D.Unsigned dt

let value (D.Any dt) e =
  match (D.kind dt, D.bits dt) with
  | Boolean, _ -> if e.[0] = '\000' then 0L else 1L
  | Signed, 4 -> Int64.of_int ((Char.code e.[0] land 15 lxor 8) - 8)
  | Unsigned, 4 -> Int64.of_int (Char.code e.[0] land 15)
  | Signed, 8 -> Int64.of_int (String.get_int8 e 0)
  | Unsigned, 8 -> Int64.of_int (String.get_uint8 e 0)
  | Signed, 16 -> Int64.of_int (String.get_int16_ne e 0)
  | Unsigned, 16 -> Int64.of_int (String.get_uint16_ne e 0)
  | Signed, 32 -> Int64.of_int32 (String.get_int32_ne e 0)
  | Unsigned, 32 ->
      Int64.logand (Int64.of_int32 (String.get_int32_ne e 0)) 0xFFFF_FFFFL
  | (Signed | Unsigned), _ -> String.get_int64_ne e 0
  | (Float | Complex), _ -> invalid_arg "value"

let order d a b =
  if unsigned d then Int64.unsigned_compare a b else Int64.compare a b

(* The element of [d] of value [x], within its range. *)
let element_of (D.Any dt) x =
  let f : type v s. (v, s) D.t -> string = function
    | D.Int64 -> P.bits D.Int64 x
    | D.Uint64 -> P.bits D.Uint64 x
    | D.Int32 -> P.bits D.Int32 (Int64.to_int32 x)
    | D.Uint32 -> P.bits D.Uint32 (Int64.to_int32 x)
    | D.Int16 -> P.bits D.Int16 (Int64.to_int x)
    | D.Uint16 -> P.bits D.Uint16 (Int64.to_int x)
    | D.Int8 -> P.bits D.Int8 (Int64.to_int x)
    | D.Uint8 -> P.bits D.Uint8 (Int64.to_int x)
    | D.Int4 -> P.bits D.Int4 (Int64.to_int x)
    | D.Uint4 -> P.bits D.Uint4 (Int64.to_int x)
    | D.Bool -> String.make 1 (Char.chr (Int64.to_int x land 255))
    | D.Bit -> P.bits D.Bit (x <> 0L)
    | D.Float32 -> P.bits D.Float32 (Int64.to_float x)
    | dt -> invalid_arg (D.name dt)
  in
  f dt

(* [d]'s least and greatest values. *)
let extremes (D.Any dt as d) =
  match (D.kind dt, D.bits dt) with
  | Boolean, _ -> (0L, 1L)
  | Unsigned, 64 -> (0L, -1L)
  | Unsigned, w -> (0L, Int64.pred (Int64.shift_left 1L w))
  | Signed, w ->
      let m = Int64.shift_left 1L (w - 1) in
      (Int64.neg m, Int64.pred m)
  | (Float | Complex), _ -> invalid_arg (name d)

let ints =
  D.
    [
      Any Int8;
      Any Uint8;
      Any Int16;
      Any Uint16;
      Any Int32;
      Any Uint32;
      Any Int64;
      Any Uint64;
      Any Int4;
      Any Uint4;
    ]

let booleans = [ boolean; bit ]
let is_bool (D.Any dt) = D.is D.Boolean dt

(* A value of [d]: small, at or beside an end, or anywhere; a bool's byte any
   byte, true where it is not 0. *)
let draw_value d =
  let open Gen in
  if same d boolean then of_list [ 0L; 0L; 1L; 1L; 2L; 128L; 255L ]
  else
    let lo, hi = extremes d in
    let clamp x =
      if order d x lo < 0 then lo else if order d x hi > 0 then hi else x
    in
    frequency
      [
        (3, map (fun x -> clamp (Int64.of_int x)) (int_range (-4) 4));
        (2, of_list [ lo; Int64.succ lo; Int64.pred hi; hi ]);
        ( 1,
          if unsigned d && hi = -1L then int64
          else if unsigned d then int64_range 0L hi
          else int64_range lo hi );
      ]

(* The integer dtype of [d]'s width and the other signedness. *)
let flipped (D.Any dt) =
  match dt with
  | D.Int64 -> D.Any D.Uint64
  | D.Uint64 -> D.Any D.Int64
  | D.Int32 -> D.Any D.Uint32
  | D.Uint32 -> D.Any D.Int32
  | D.Int16 -> D.Any D.Uint16
  | D.Uint16 -> D.Any D.Int16
  | D.Int8 -> D.Any D.Uint8
  | D.Uint8 -> D.Any D.Int8
  | D.Int4 -> D.Any D.Uint4
  | D.Uint4 -> D.Any D.Int4
  | dt -> invalid_arg (D.name dt)

(* A ranged program's tree, each node of the dtype its maker gives it, over
   operand 0 of float32 and the loads [loads], whose operand [k + 1] is
   [loads.(k)]. *)
type tree =
  | Leaf of P.node * D.any
  | T1 of P.op1 * D.any * tree
  | T2 of P.op2 * tree * tree
  | T3 of P.op3 * tree * tree * tree

let rec dtype_of_tree = function
  | Leaf (_, d) | T1 (_, d, _) -> d
  | T2 (Compare _, _, _) -> boolean
  | T2 (Binary _, x, _) -> dtype_of_tree x
  | T3 (_, _, x, _) -> dtype_of_tree x

let const d x = Leaf (P.Const (d, element_of d x), d)

(* A leaf of [d]: a coordinate, a constant, or a load, each cast to [d] where it
   is of another dtype. *)
let leaf loads (D.Any dt as d) =
  let open Gen in
  let cast l = if same (dtype_of_tree l) d then l else T1 (Cast, d, l) in
  let load k = Leaf (P.In (k + 1), loads.(k)) in
  if D.is D.Float dt then constant (Leaf (P.In 0, f32))
  else
    let* pick = int_range 0 3 in
    match pick with
    | 0 ->
        let+ c = int_range 0 2 in
        cast (Leaf (P.Coord c, i64))
    | 1 -> map (const d) (draw_value d)
    | _ ->
        let+ k = int_range 0 (Array.length loads - 1) in
        cast (load k)

let rec tree loads depth (D.Any dt as d) =
  let open Gen in
  let at = tree loads (depth - 1) in
  let leaf () = leaf loads d in
  if depth <= 0 then leaf ()
  else
    match D.kind dt with
    | Boolean ->
        (* A comparison's bool, cast where [d] is bit. *)
        let as_d t = if same d boolean then t else T1 (Cast, d, t) in
        let* k = int_range 0 7 in
        if k = 0 then leaf ()
        else if k <= 3 then
          let* x = of_list ~pp:pp_dtype ((i64 :: i64 :: ints) @ booleans) in
          let* c = of_list P.[ Less; Less_equal; Equal; Not_equal ] in
          let+ a = at x and+ b = at x in
          as_d (T2 (Compare c, a, b))
        else if k = 4 then
          let* b = of_list P.[ And; Or; Xor ] in
          let+ x = at d and+ y = at d in
          T2 (Binary b, x, y)
        else if k = 5 then
          let* x = of_list ~pp:pp_dtype (ints @ booleans) in
          let+ a = at x in
          T1 (Cast, d, a)
        else if k = 6 then
          let+ a = at f32 and+ b = at f32 in
          as_d (T2 (Compare Less, a, b))
        else
          let+ c = at d and+ x = at d and+ y = at d in
          T3 (Where, c, x, y)
    | Float ->
        let* k = int_range 0 2 in
        if k = 0 then constant (Leaf (P.In 0, f32))
        else if k = 1 then
          let* x = of_list ~pp:pp_dtype ints in
          let+ a = at x in
          T1 (Cast, d, a)
        else
          let+ a = at d and+ b = at d in
          T2 (Binary Add, a, b)
    | Signed | Unsigned ->
        let* k = int_range 0 14 in
        let neg = if unsigned d then 3L else -3L in
        if k <= 1 then leaf ()
        else if k <= 4 then
          let* b = of_list P.[ Add; Sub; Mul; Maximum; Minimum ] in
          let+ x = at d and+ y = at d in
          T2 (Binary b, x, y)
        else if k = 5 then
          let* m = map Int64.of_int (int_range (-5) (-1)) in
          let m = if unsigned d then Int64.neg m else m in
          let+ x = at d in
          T2 (Binary Mul, x, const d m)
        else if k = 6 then
          let* b = of_list P.[ Idiv; Mod ] in
          let* by =
            frequency
              [
                (1, at d);
                (1, constant (const d neg));
                (1, constant (const d 4L));
              ]
          in
          let+ x = at d in
          T2 (Binary b, x, by)
        else if k = 7 then
          let* u = of_list P.[ Neg; Abs; Sign; Recip ] in
          let+ x = at d in
          T1 (Unary u, d, x)
        else if k <= 9 then
          let* x = of_list ~pp:pp_dtype (booleans @ ints) in
          let+ a = at x in
          T1 (Cast, d, a)
        else if k = 10 then
          let+ a = at f32 in
          T1 (Cast, d, a)
        else if k = 11 then
          let+ c = at boolean and+ x = at d and+ y = at d in
          T3 (Where, c, x, y)
        else if k = 12 then
          let+ x = at d and+ y = at d and+ z = at d in
          T3 (Fma, x, y, z)
        else if k = 13 then
          let* b = of_list P.[ And; Xor; Pow ] in
          let+ x = at d and+ y = at d in
          T2 (Binary b, x, y)
        else
          let byte = D.bits dt = 8 in
          let* s =
            of_list ~pp:pp_dtype
              (flipped d :: (if byte then [ boolean ] else []))
          in
          let+ x = at s in
          T1 (Bitcast, d, x)
    | Complex -> leaf ()

(* A case: the loads' dtypes, the program's nodes, the box's extents and
   coordinates' offsets, and each load's elements. *)
type ranged_case = {
  loads : D.any array;
  nodes : P.node array;
  shape : int array;
  offsets : int64 array;
  values : int64 array array;
}

let lower t =
  let nodes = ref [] and count = ref 0 in
  let add n =
    nodes := n :: !nodes;
    incr count;
    !count - 1
  in
  let rec go = function
    | Leaf (n, _) -> add n
    | T1 (k, d, x) ->
        let x = go x in
        add (P.Op1 (k, d, x))
    | T2 (k, x, y) ->
        let x = go x in
        let y = go y in
        add (P.Op2 (k, x, y))
    | T3 (k, x, y, z) ->
        let x = go x in
        let y = go y in
        let z = go z in
        add (P.Op3 (k, x, y, z))
  in
  ignore (go t : int);
  Array.of_list (List.rev !nodes)

let ranged_ins c = Array.append [| f32 |] c.loads

(* The program of every integer and boolean node of [c] as an output. *)
let all_ranged c =
  let p =
    P.v
      ~ins:(Iarray.of_array (ranged_ins c))
      (Iarray.of_array c.nodes) ~outs:[| 0 |]
  in
  let outs =
    List.filter
      (fun i ->
        let (D.Any dt) = P.dtype p i in
        not (D.is D.Float dt || D.is D.Complex dt))
      (List.init (Array.length c.nodes) Fun.id)
  in
  P.v ~ins:p.ins p.nodes ~outs:(Iarray.of_list outs)

let pp_ranged ppf c =
  let p = all_ranged c in
  Format.fprintf ppf "loads [%s]@ shape %a@ offsets [%s]@ %a"
    (String.concat "; " (Array.to_list (Array.map name c.loads)))
    Elements.pp_ints c.shape
    (String.concat "; " (Array.to_list (Array.map Int64.to_string c.offsets)))
    pp_prog p;
  Array.iteri
    (fun k v ->
      Format.fprintf ppf "@ load %d [%s]" k
        (String.concat "; " (Array.to_list (Array.map Int64.to_string v))))
    c.values

(* Offsets that put the coordinates near the ends of each width. *)
let coordinate_offsets =
  Gen.of_list
    [
      0L;
      0L;
      5L;
      125L;
      254L;
      32765L;
      65533L;
      0x7FFF_FFFDL;
      0xFFFF_FFFDL;
      0x7FFF_FFFF_FFFF_FFF0L;
    ]

let ranged_cases =
  let open Gen in
  with_pp pp_ranged
    (let* loads =
       array ~size:(int_range 1 3) (of_list ~pp:pp_dtype (booleans @ ints))
     in
     let* out = of_list ~pp:pp_dtype (booleans @ booleans @ ints) in
     let* t = tree loads 3 out in
     let* shape = array ~size:(constant 3) (int_range 1 4) in
     let* offsets = array ~size:(constant 3) coordinate_offsets in
     let n = Array.fold_left ( * ) 1 shape in
     let values d =
       frequency
         [
           (1, map (Array.make n) (draw_value d));
           (3, array ~size:(constant n) (draw_value d));
         ]
     in
     let rec each = function
       | [] -> constant []
       | d :: rest ->
           let* v = values d in
           let+ vs = each rest in
           v :: vs
     in
     let+ values = each (Array.to_list loads) in
     { loads; nodes = lower t; shape; offsets; values = Array.of_list values })

let elements1 d x = operand1 d (element_of d x)

(* The box of [c]: the least and greatest of each load and coordinate, the
   float32 operand's as 0. *)
let bounds c =
  Array.concat
    [
      [| elements1 f32 0L; elements1 f32 0L |];
      Array.concat
        (List.mapi
           (fun k d ->
             let v = c.values.(k) in
             let pick f =
               Array.fold_left
                 (fun a x -> if f (order d x a) then x else a)
                 v.(0) v
             in
             let least = pick (fun o -> o < 0)
             and most = pick (fun o -> o > 0) in
             [| elements1 d least; elements1 d most |])
           (Array.to_list c.loads));
      Array.concat
        (List.init 3 (fun a ->
             let lo = c.offsets.(a) in
             let hi = Int64.add lo (Int64.of_int (c.shape.(a) - 1)) in
             [| elements1 i64 lo; elements1 i64 hi |]));
    ]

let refused what = function
  | Ok r -> r
  | Error a -> failf "%s refused: %a" what Nx_array_support.pp_answer a

(* Every integer and boolean node of a drawn program lies, at every index of the
   box, within its range over the box. *)
let law_ranges c =
  let p = all_ranged c in
  let n = Array.fold_left ( * ) 1 c.shape in
  let loads =
    Array.append
      [| Elements.of_elements f32 c.shape (Array.make n (element_of f32 1L)) |]
      (Array.mapi
         (fun k d ->
           Elements.of_elements d c.shape
             (Array.map (element_of d) c.values.(k)))
         c.loads)
  in
  let shifted =
    P.subst ~ins:p.ins
      (function
        | Coord a when c.offsets.(a) <> 0L ->
            Some P.E.(any (coord a + const D.Int64 c.offsets.(a)))
        | _ -> None)
      p
  in
  let values = refused "the program" (compute ~shape:c.shape shifted loads) in
  let q = P.ranges ~axes:3 p in
  let r = refused "the ranges" (compute q (bounds c)) in
  let outs = Iarray.to_array p.outs in
  let at = Hashtbl.create 16 in
  Array.iteri (fun o i -> Hashtbl.replace at i o) outs;
  let vals i = Array.map (value (P.dtype p i)) values.(Hashtbl.find at i) in
  let range i =
    let o = Hashtbl.find at i in
    let d = P.dtype p i in
    (value d r.(2 * o).(0), value d r.((2 * o) + 1).(0))
  in
  let ranged i = Hashtbl.mem at i in
  (* What the case reaches. *)
  let all f a = Array.for_all f a and exists f a = Array.exists f a in
  let node i = Iarray.get p.nodes i in
  let bools =
    List.filter (fun i -> is_bool (P.dtype p i)) (Array.to_list outs)
  in
  cover "a boolean true at every index by its range"
    (List.exists (fun i -> fst (range i) = 1L) bools);
  cover "a boolean false at every index by its range"
    (List.exists (fun i -> snd (range i) = 0L) bools);
  cover "a boolean unknown by its range"
    (List.exists (fun i -> range i = (0L, 1L)) bools);
  let fits d x =
    let lo, hi = extremes d in
    order d x lo >= 0 && order d x hi <= 0
  in
  let narrow (D.Any dt) = D.bits dt <= 16 in
  let wraps i =
    match node i with
    | Op2 (Binary ((Add | Sub | Mul) as b), x, y)
      when narrow (P.dtype p i) && ranged x ->
        let f =
          match b with Add -> Int64.add | Sub -> Int64.sub | _ -> Int64.mul
        in
        let d = P.dtype p i in
        let xs = vals x and ys = vals y in
        exists (fun j -> not (fits d (f xs.(j) ys.(j)))) (Array.init n Fun.id)
    | _ -> false
  in
  cover "an addition, subtraction or product that wraps in the box"
    (exists wraps outs);
  (* Whether [v], an element of [s], is outside [d]'s range. *)
  let outside s d v =
    let lo, hi = extremes d in
    if (not (unsigned s)) && v < 0L then unsigned d || v < lo
    else if unsigned s && v < 0L then hi <> -1L || not (unsigned d)
    else (not (hi = -1L && unsigned d)) && v > hi
  in
  let cast_unfit i =
    match node i with
    | Op1 (Cast, d, x) when ranged x && not (is_bool (P.dtype p x) || is_bool d)
      ->
        exists (outside (P.dtype p x) d) (vals x)
    | _ -> false
  in
  cover "a cast whose operand does not fit its dtype" (exists cast_unfit outs);
  let negative_multiplier i =
    match node i with
    | Op2 (Binary Mul, x, y) when ranged x && not (unsigned (P.dtype p i)) ->
        let neg k = all (fun v -> v < 0L) (vals k) in
        let varies k = exists (fun v -> v <> (vals k).(0)) (vals k) in
        (neg x && varies y) || (neg y && varies x)
    | _ -> false
  in
  cover "a negative multiplier of a range" (exists negative_multiplier outs);
  let unknown_where i =
    match node i with
    | Op3 (Where, c, _, _) when ranged i -> range c = (0L, 1L)
    | _ -> false
  in
  cover "a where on an unknown condition" (exists unknown_where outs);
  let from_float i =
    match node i with
    | Op1 (_, _, x) | Op2 (_, x, _) -> P.dtype p x = f32
    | _ -> false
  in
  cover "an integer or boolean computed from a float" (exists from_float outs);
  Array.iter
    (fun i ->
      let d = P.dtype p i in
      let lo, hi = range i in
      Array.iteri
        (fun j x ->
          if order d x lo < 0 || order d x hi > 0 then
            failf "node %d (%s) is %Ld at index %d, outside [%Ld, %Ld]" i
              (name d) x j lo hi)
        (vals i))
    outs

(* The range of [p]'s one output over [box]: its least and greatest elements. *)
let range_of ~axes p box =
  let r = refused "the ranges" (compute (P.ranges ~axes p) box) in
  (r.(0).(0), r.(1).(0))

let ends (D.Any dt) =
  match D.kind dt with
  | Boolean -> (P.bits dt false, P.bits dt true)
  | _ ->
      let lo, hi = extremes (D.Any dt) in
      (element_of (D.Any dt) lo, element_of (D.Any dt) hi)

(* The program of [nodes] over [ins] whose output is its last node. *)
let last_out ins nodes =
  P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
    ~outs:[| Array.length nodes - 1 |]

(* Each row: a program, the axes of its box, the box, and the output's dtype,
   whose whole range is its range (prog.mli: "the range is the dtype's whole
   range"). *)
let wrapping =
  let i8 = D.Any D.Int8 and u8 = D.Any D.Uint8 and p = last_out in
  [
    ( "an int8 add whose greatest sum passes 127",
      0,
      p [| i8 |]
        [| In 0; Const (i8, P.bits D.Int8 10); Op2 (Binary Add, 0, 1) |],
      [| elements1 i8 100L; elements1 i8 120L |],
      i8 );
    ( "a uint8 sub whose least difference is below 0",
      0,
      p [| u8 |]
        [| Const (u8, P.bits D.Uint8 0); In 0; Op2 (Binary Sub, 0, 1) |],
      [| elements1 u8 0L; elements1 u8 1L |],
      u8 );
    ( "an int8 cast of a coordinate up to 200",
      1,
      p [||] [| Coord 0; Op1 (Cast, i8, 0) |],
      [| elements1 i64 0L; elements1 i64 200L |],
      i8 );
  ]

let from_floats =
  let p = last_out in
  [
    ( "an int32 cast of a float32",
      0,
      p [| f32 |] [| In 0; Op1 (Cast, i32, 0) |],
      [| elements1 f32 1L; elements1 f32 2L |],
      i32 );
    ( "a comparison of float32s",
      0,
      p [| f32; f32 |] [| In 0; In 1; Op2 (Compare Less, 0, 1) |],
      [|
        elements1 f32 1L; elements1 f32 1L; elements1 f32 2L; elements1 f32 2L;
      |],
      boolean );
  ]

let test_whole_range (_, axes, p, box, d) =
  equal (pair string string) (ends d) (range_of ~axes p box)

(* A cast into bit over a coordinate's bounds [lo, hi]: [Cast Bit] of the
   coordinate, and of it with [true]. The range is that of [x <> 0] over [lo,
   hi], of bit's dtype. *)
let bit_casts =
  [
    ("zero", (0L, 0L), ("\000", "\000"));
    ("apart from zero", (1L, 3L), ("\001", "\001"));
    ("across zero", (0L, 2L), ("\000", "\001"));
  ]

let test_bit_cast (_, (lo, hi), want) =
  let p =
    P.v ~ins:[||]
      [|
        P.Coord 0;
        Op1 (Cast, bit, 0);
        Const (bit, P.bits D.Bit true);
        Op2 (Binary And, 1, 2);
      |]
      ~outs:[| 3 |]
  in
  equal (pair string string) want
    (range_of ~axes:1 p [| elements1 i64 lo; elements1 i64 hi |])

let law_no_float_bounds c =
  let p = all_ranged c in
  cover "a node reads the float operand"
    (Iarray.exists (function P.In 0 -> true | _ -> false) p.nodes);
  let q = P.ranges ~axes:3 p in
  let reads =
    List.filter_map
      (function P.In i when i < 2 -> Some i | _ -> None)
      (Iarray.to_list q.nodes)
  in
  equal (list int) [] reads

let law_range_operands c =
  let q = P.ranges ~axes:3 (all_ranged c) in
  let bounds d = [ name d; name d ] in
  equal (list string)
    (List.concat_map bounds (Array.to_list (ranged_ins c))
    @ List.concat_map bounds [ i64; i64; i64 ])
    (List.map name (Iarray.to_list q.ins))

let ranged_picks =
  let open Gen in
  with_pp
    (fun ppf (c, sel) ->
      Format.fprintf ppf "%a,@ picking %a" pp_ranged c Elements.pp_ints sel)
    (let+ c = ranged_cases
     and+ sel = array ~size:(int_range 1 4) (int_range 0 1_000) in
     let n = Iarray.length (all_ranged c).outs in
     (c, Array.map (fun k -> k mod n) sel))

let law_range_outs (c, sel) =
  let p = all_ranged c in
  let n = Array.length sel in
  cover "an output picked twice"
    (List.length (List.sort_uniq Int.compare (Array.to_list sel)) < n);
  cover "outputs reordered"
    (List.exists (fun j -> sel.(j) > sel.(j + 1)) (List.init (n - 1) Fun.id));
  let all = refused "the ranges" (compute (P.ranges ~axes:3 p) (bounds c)) in
  let want =
    Array.concat
      (List.map
         (fun k -> [| all.(2 * k); all.((2 * k) + 1) |])
         (Array.to_list sel))
  in
  equal outputs (Ok want)
    (compute (P.ranges ~axes:3 ~outs:(Iarray.of_array sel) p) (bounds c))

let test_range_axes () =
  let p = P.v ~ins:[| i32 |] [| In 0 |] ~outs:[| 0 |] in
  List.iter
    (fun axes ->
      equal ~msg:(strf "axes %d" axes) int
        ((2 * axes) + 2)
        (Iarray.length (P.ranges ~axes p).ins))
    (List.init (Nx_array.Layout.max_rank + 1) Fun.id)

let range_refusals =
  let p = last_out in
  let ints = p [| i32 |] [| In 0 |] in
  [
    ("negative axes", fun () -> P.ranges ~axes:(-1) ints);
    ( "axes past max_rank",
      fun () -> P.ranges ~axes:(Nx_array.Layout.max_rank + 1) ints );
    ( "an output reading the coordinate of axis axes",
      fun () -> P.ranges ~axes:2 (p [||] [| Coord 2 |]) );
    ("a float32 output", fun () -> P.ranges ~axes:0 (p [| f32 |] [| In 0 |]));
    ( "a complex64 output",
      fun () -> P.ranges ~axes:0 (p [| D.Any D.Complex64 |] [| In 0 |]) );
    ("no output", fun () -> P.ranges ~axes:0 ~outs:[||] ints);
    ("an output past p's", fun () -> P.ranges ~axes:0 ~outs:[| 1 |] ints);
    ("a negative output", fun () -> P.ranges ~axes:0 ~outs:[| -1 |] ints);
  ]

let test_range_refuses (_, f) = refuses f

let tests =
  [
    group "programs"
      [
        test
          "the Box-Muller example is one node per distinct subexpression, in \
           the walk's finish order"
          test_box_muller;
        prop "equal programs are equal and hash alike" drawn law_equal_programs;
      ];
    group "domains"
      [
        cases "each kind takes the dtypes its domain lists" ~id:kind_name kinds
          test_domain;
      ];
    group "bits"
      [
        prop ~count:200
          ~examples:(List.map (fun w -> (w, w)) special_words)
          "bits dt x is the bytes of a one-element array of x, at every dtype"
          (Gen.pair word word) law_bits;
        cases "a NaN stores quiet, of its sign, with the top of its payload"
          ~id:(fun (x, _, _) -> strf "%016Lx" x)
          nan_stores test_nan_bits;
        cases "an int outside the dtype's range raises, its ends do not"
          ~id:(fun (Ints (n, _, _, _)) -> n)
          int_dtypes test_int_range;
      ];
    group "v"
      [
        prop "v keeps its operands, nodes and outputs" drawn law_v;
        cases
          "v takes a kind's node exactly where accepts1, accepts2 or accepts3 \
           takes its dtypes"
          ~id:kind_name kinds test_v_kinds;
        cases "v raises Invalid_argument for an ill-formed program"
          ~id:(fun (n, _, _, _) -> n)
          ill_formed test_ill_formed;
        cases
          "a Const of bool takes any byte and of bit 0 or 1, true iff not zero"
          ~id:(fun (n, _, _, _) -> n)
          bool_consts test_bool_const;
        test "a program takes any number of operands and outputs"
          test_many_operands;
        test "v checks a program allocating only its nodes' dtypes"
          test_v_allocates;
        prop "of_exprs and subst make programs that hold as v states"
          (Gen.pair drawn offsets) law_holds_as_v;
      ];
    group "of_node"
      [
        prop
          "of_node ~ins n is v over In 0 to In (k - 1) then n, raising where v \
           does"
          one_node law_of_node;
      ];
    group "dtype"
      [
        prop "dtype is each node's as dtype states" drawn law_dtype;
        cases "dtype raises for a node past the program or negative"
          ~id:string_of_int [ 1; -1 ] test_dtype_refuses;
      ];
    group "is_identity"
      [
        prop ~examples:near_identities
          "is_identity is true iff one operand, one output, its In" drawn
          law_identity;
      ];
    group "single"
      [
        prop ~examples:singles
          "single of of_node's program is its node iff the node reads its \
           operands in order"
          one_node law_single;
        cases "single is None for a program of_node does not make" ~id:fst
          not_of_node test_not_single;
      ];
    group "is_single"
      [
        prop ~examples:is_single_examples
          "is_single is true iff single gives the last node" drawn law_is_single;
        test "is_single allocates nothing" test_is_single_allocates;
      ];
    group "E"
      [
        test "dtype is the dtype of the expression's node" test_expr_dtypes;
        cases "each kind function makes its kind's node"
          ~id:(fun (n, _, _) -> n)
          kind_functions test_kind_function;
        test "cast into an expression's own dtype is the expression"
          test_cast_own_dtype;
        cases "each operator makes the node of the function it names"
          ~id:(fun (n, _, _, _) -> n)
          operators test_operator;
        test "a constructor raises where v refuses its node" test_constructors;
        cases "a kind function of one operand raises where v refuses its node"
          ~id:(fun r -> unary_name r.unary)
          unary_fns test_unary_fn;
        cases "a kind function of two operands raises where v refuses its node"
          ~id:(fun r -> binary_name r.binary)
          binary_fns test_binary_fn;
        cases "input, coord and const raise as stated" ~id:fst leaf_refusals
          test_leaf_refuses;
      ];
    group "of_exprs"
      [
        prop ~count:300 ~examples:[ signed_zero ]
          "of_exprs's nodes are the outputs' distinct subexpressions once, in \
           depth-first finish order, whichever builds made them"
          drawn law_of_exprs;
        test "of_exprs merges two builds of 15000 nodes"
          test_of_exprs_two_builds;
        test "of_exprs merges 2^17 fresh copies into 17 nodes"
          test_of_exprs_fresh_copies;
        cases
          "of_exprs raises for no output, an In past ins or of another dtype"
          ~id:(fun (n, _, _) -> n)
          of_exprs_refusals test_of_exprs_refuses;
      ];
    group "subst"
      [
        prop
          "subst is of_exprs of the outputs with each reached leaf replaced by \
           f's expression"
          (Gen.pair drawn offsets) law_subst;
        prop
          "subst calls f once on each leaf the picked outputs reach, and on \
           nothing else"
          selecting law_subst_calls;
        prop
          "subst replacing In k by an expression over the other operands \
           lowers p over it"
          binding law_bind;
        cases "subst binds an operand to an operand, a coordinate or a constant"
          ~id:(fun (n, _, _, _) -> n)
          bind_leaves test_bind_leaf;
        prop "subst with outs is the outputs of p it picks, in order" selecting
          law_subst_outs;
        test "a leaf f replaces needs no operand of ins" test_replaced_leaf;
        test
          "a leaf the picked outputs do not reach is not passed to f and needs \
           no operand of ins"
          test_unreached_leaf;
        cases "subst raises where its expressions or outs do not fit"
          ~id:(fun (n, _, _, _) -> n)
          subst_refusals test_subst_refuses;
      ];
    group "ranges"
      [
        prop ~count:400 "every output lies within its range over the box"
          ranged_cases law_ranges;
        cases "a kind that may wrap has its dtype's whole range"
          ~id:(fun (n, _, _, _, _) -> n)
          wrapping test_whole_range;
        cases "a node computed from a float has its dtype's whole range"
          ~id:(fun (n, _, _, _, _) -> n)
          from_floats test_whole_range;
        cases "a cast into bit has bit's range"
          ~id:(fun (n, _, _) -> n)
          bit_casts test_bit_cast;
        prop "no node of ranges reads a float operand's bounds" ranged_cases
          law_no_float_bounds;
        prop
          "ranges's operands are each operand's least and greatest, then each \
           axis's"
          ranged_cases law_range_operands;
        prop
          "ranges ~outs picks the kth output's range as outputs 2k and 2k + 1"
          ranged_picks law_range_outs;
        test "ranges takes axes from 0 to max_rank" test_range_axes;
        cases "ranges raises for axes, outputs or outs it cannot range" ~id:fst
          range_refusals test_range_refuses;
      ];
  ]

let () = exit (run "nx_kernel.prog" tests)
