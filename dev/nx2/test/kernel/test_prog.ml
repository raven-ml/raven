(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Programs: the kinds' domains against the table prog.mli states, constant
   bits against an array's bytes, and programs against what they were made
   of and what C reads. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog

let strf = Printf.sprintf
let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let name (D.Any dt) = D.name dt

(* Domains *)

let unaries =
  P.
    [
      Neg; Recip; Abs; Sign; Sqrt; Exp; Exp2; Log; Log2; Log1p; Expm1; Sin;
      Cos; Tan; Asin; Acos; Atan; Sinh; Cosh; Tanh; Erf; Floor; Ceil; Round;
      Trunc;
    ]

let binaries =
  P.
    [
      Add; Sub; Mul; Fdiv; Idiv; Mod; Pow; Atan2; Maximum; Minimum; And; Or;
      Xor; Threefry;
    ]

let compares = P.[ Equal; Not_equal; Less; Less_equal ]

let op1s =
  P.[ Copy; Cast; Bitcast ] @ List.map (fun u -> P.Unary u) unaries

let op2s =
  List.map (fun b -> P.Binary b) binaries
  @ List.map (fun c -> P.Compare c) compares

(* The kind names nx_kinds.h and the C reader use. *)
let unary_name : P.unary -> string = function
  | Neg -> "neg"
  | Recip -> "recip"
  | Abs -> "abs"
  | Sign -> "sign"
  | Sqrt -> "sqrt"
  | Exp -> "exp"
  | Exp2 -> "exp2"
  | Log -> "log"
  | Log2 -> "log2"
  | Log1p -> "log1p"
  | Expm1 -> "expm1"
  | Sin -> "sin"
  | Cos -> "cos"
  | Tan -> "tan"
  | Asin -> "asin"
  | Acos -> "acos"
  | Atan -> "atan"
  | Sinh -> "sinh"
  | Cosh -> "cosh"
  | Tanh -> "tanh"
  | Erf -> "erf"
  | Floor -> "floor"
  | Ceil -> "ceil"
  | Round -> "round"
  | Trunc -> "trunc"

let binary_name : P.binary -> string = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Fdiv -> "fdiv"
  | Idiv -> "idiv"
  | Mod -> "mod"
  | Pow -> "pow"
  | Atan2 -> "atan2"
  | Maximum -> "maximum"
  | Minimum -> "minimum"
  | And -> "and"
  | Or -> "or"
  | Xor -> "xor"
  | Threefry -> "threefry"

let compare_name : P.compare -> string = function
  | Equal -> "equal"
  | Not_equal -> "not_equal"
  | Less -> "less"
  | Less_equal -> "less_equal"

let op1_name = function
  | P.Copy -> "copy"
  | Cast -> "cast"
  | Bitcast -> "bitcast"
  | Unary u -> unary_name u

let op2_name = function
  | P.Binary b -> binary_name b
  | Compare c -> compare_name c

let op3_name = function P.Where -> "where" | Fma -> "fma"

(* prog.mli's table, by kind of number. *)
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
  | Neg | Recip -> numbers
  | Abs | Sign | Floor | Ceil | Round | Trunc -> reals
  | Sqrt | Exp | Exp2 | Log | Log2 | Log1p | Expm1 | Sin | Cos | Tan | Asin
  | Acos | Atan | Sinh | Cosh | Tanh | Erf ->
      [ Float ]

let binary_domain : P.binary -> cls list = function
  | Add | Sub | Mul -> numbers
  | Fdiv -> [ Float; Complex ]
  | Idiv -> integers
  | Mod | Pow -> reals
  | Atan2 -> [ Float ]
  | Maximum | Minimum -> every
  | And | Or | Xor -> Boolean :: integers
  | Threefry -> []

let same (D.Any x) (D.Any y) = D.equal x y
let bits (D.Any dt) = D.bits dt

let want1 k x y =
  match k with
  | P.Copy -> same x y
  | Cast -> true
  | Bitcast -> bits x = bits y
  | Unary u -> same x y && List.mem (cls x) (unary_domain u)

let want2 k x =
  match k with
  | P.Binary Threefry -> name x = "uint64"
  | Binary b -> List.mem (cls x) (binary_domain b)
  | Compare _ -> true

let want3 k c x =
  match k with
  | P.Where -> cls c = Boolean
  | Fma -> same c x && List.mem (cls x) reals

(* Whether [of_node] builds the program of [n] over operands of [ins]. *)
let builds ins n =
  match P.of_node ~ins:(Iarray.of_array ins) n with
  | _ -> true
  | exception Invalid_argument _ -> false

let test_domains () =
  let dts = D.all in
  List.iter
    (fun (D.Any x as ax) ->
      equal
        ~msg:(strf "fill %s" (name ax))
        bool true
        (P.accepts0 (Fill "") x);
      equal
        ~msg:(strf "iota %s" (name ax))
        bool
        (List.mem (cls ax) reals)
        (P.accepts0 (Iota 0) x);
      List.iter
        (fun k ->
          equal
            ~msg:(strf "%s %s" (op2_name k) (name ax))
            bool (want2 k ax) (P.accepts2 k x);
          equal
            ~msg:(strf "node %s %s" (op2_name k) (name ax))
            bool (want2 k ax)
            (builds [| ax; ax |] (Op2 (k, 0, 1))))
        op2s;
      List.iter
        (fun (D.Any y as ay) ->
          List.iter
            (fun k ->
              equal
                ~msg:(strf "%s %s to %s" (op1_name k) (name ax) (name ay))
                bool (want1 k ax ay) (P.accepts1 k x y);
              equal
                ~msg:(strf "node %s %s to %s" (op1_name k) (name ax) (name ay))
                bool (want1 k ax ay)
                (builds [| ax |] (Op1 (k, ay, 0))))
            op1s;
          List.iter
            (fun k ->
              equal
                ~msg:(strf "%s %s %s" (op3_name k) (name ax) (name ay))
                bool (want3 k ax ay) (P.accepts3 k x y);
              equal
                ~msg:(strf "node %s %s %s" (op3_name k) (name ax) (name ay))
                bool (want3 k ax ay)
                (builds [| ax; ay; ay |] (Op3 (k, 0, 1, 2))))
            P.[ Where; Fma ])
        dts)
    dts;
  equal ~msg:"two dtypes" bool false
    (builds [| D.Any D.Float32; D.Any D.Float64 |] (Op2 (Binary Add, 0, 1)))

(* Each kind a base dtype takes has its function in nx_kinds.h at the dtype's
   compute type. Floor, Ceil, Round and Trunc are the identity on integers,
   which no function computes. *)
let test_domains_computed () =
  let base =
    D.
      [
        Any Float32; Any Float64; Any Int8; Any Uint8; Any Int16; Any Uint16;
        Any Int32; Any Uint32; Any Int64; Any Uint64; Any Bool;
      ]
  in
  let compute (D.Any dt as d) =
    match (cls d, D.bits dt) with
    | Float, 64 -> `F64
    | Float, _ -> `F32
    | Signed, 64 -> `Int "i64"
    | Signed, _ -> `Int "i32"
    | Unsigned, 64 -> `Int "u64"
    | (Unsigned | Boolean), _ -> `Int "u32"
    | Complex, _ -> invalid_arg "no compute type"
  in
  let calls kind arity d =
    match compute d with
    | `F32 -> ignore (Nx_kinds_support.f32 kind (Array.make arity 0))
    | `F64 -> ignore (Nx_kinds_support.f64 kind (Array.make arity 0.))
    | `Int ty -> ignore (Nx_kinds_support.int ty kind (Array.make arity 0L))
  in
  let identity = P.[ Floor; Ceil; Round; Trunc ] in
  List.iter
    (fun (D.Any x as d) ->
      List.iter
        (fun u ->
          if
            P.accepts1 (Unary u) x x
            && not (cls d <> Float && List.mem u identity)
          then calls (unary_name u) 1 d)
        unaries;
      List.iter
        (fun b ->
          if P.accepts2 (Binary b) x && b <> Threefry then
            calls (binary_name b) 2 d)
        binaries;
      List.iter
        (fun c -> if P.accepts2 (Compare c) x then calls (compare_name c) 2 d)
        compares;
      if P.accepts3 Fma x x then calls "fma" 3 d)
    base

let test_domains_allocate_nothing () =
  let k = P.Binary Add in
  let before = Gc.minor_words () in
  for _ = 1 to 1000 do
    ignore (Sys.opaque_identity (P.accepts2 k D.Float32))
  done;
  equal ~msg:"words" (float 0.5) 0. (Gc.minor_words () -. before)

(* Bits *)

(* An element's bits as a one-element array holds them: its bytes, or a
   sub-byte code in one byte. *)
let array_bits (type v s) (dt : (v, s) D.t) (x : v) =
  let a = A.of_array dt [| 1 |] [| x |] in
  match D.bits dt with
  | 1 -> String.make 1 (Char.chr (Bool.to_int (A.get (A.expect D.Bit (A.Any a)) [| 0 |])))
  | 4 -> String.make 1 (Char.chr (A.get (Option.get (A.bitcast D.Uint4 a)) [| 0 |]))
  | _ ->
      let b = A.to_array (Option.get (A.bitcast D.Uint8 a)) in
      String.init (Array.length b) (fun i -> Char.chr b.(i))

let hex s =
  String.concat "" (List.init (String.length s) (fun i -> strf "%02x" (Char.code s.[i])))

let law_bits (D.Any dt, f) =
  let x = D.of_float dt f in
  equal ~msg:(name (D.Any dt)) string (hex (array_bits dt x)) (hex (P.bits dt x))

let test_bits_refuse () =
  raises_match ~msg:"int8" Exn.invalid_arg (fun () -> P.bits D.Int8 128);
  raises_match ~msg:"uint4" Exn.invalid_arg (fun () -> P.bits D.Uint4 16);
  raises_match ~msg:"int16" Exn.invalid_arg (fun () -> P.bits D.Int16 (-32769))

(* Programs *)

(* A program drawn from operand dtypes and choices: each node picks, by its
   choice, one of the nodes the earlier ones admit. *)
type drawn = {
  ins : D.any array;
  nodes : P.node array;
  types : D.any array;
  outs : int array;
}

let base_dtypes =
  D.
    [
      Any Float32; Any Float64; Any Float16; Any Int32; Any Uint8; Any Int64;
      Any Uint64; Any Bool; Any Complex64; Any Int4;
    ]

let build ins choices nouts =
  let types = ref [||] and nodes = ref [] in
  let add n dt =
    nodes := n :: !nodes;
    types := Array.append !types [| dt |]
  in
  Array.iter
    (fun c ->
      let n = Array.length !types in
      let pick l = List.nth l (c mod List.length l) in
      let earlier = List.init n Fun.id in
      let candidates =
        List.concat
          [
            List.init (Array.length ins) (fun k -> (P.In k, ins.(k)));
            [ (P.Coord (c mod 3), D.Any D.Int64) ];
            List.map
              (fun (D.Any dt as d) ->
                (P.Const (d, P.bits dt (D.of_float dt (Float.of_int (c mod 7)))), d))
              base_dtypes;
            List.concat_map
              (fun i ->
                let (D.Any x) = !types.(i) in
                List.concat_map
                  (fun (D.Any y as d) ->
                    List.filter_map
                      (fun k ->
                        if P.accepts1 k x y then Some (P.Op1 (k, d, i), d)
                        else None)
                      op1s)
                  base_dtypes)
              earlier;
            List.concat_map
              (fun i ->
                List.concat_map
                  (fun j ->
                    let (D.Any x) = !types.(i) in
                    if not (same !types.(i) !types.(j)) then []
                    else
                      List.filter_map
                        (fun k ->
                          if P.accepts2 k x then
                            Some
                              ( P.Op2 (k, i, j),
                                match k with
                                | Compare _ -> D.Any D.Bool
                                | Binary _ -> !types.(i) )
                          else None)
                        op2s)
                  earlier)
              earlier;
            List.concat_map
              (fun i ->
                List.concat_map
                  (fun j ->
                    let (D.Any c') = !types.(i) in
                    let (D.Any x) = !types.(j) in
                    List.filter_map
                      (fun k ->
                        if P.accepts3 k c' x then
                          Some (P.Op3 (k, i, j, j), !types.(j))
                        else None)
                      P.[ Where; Fma ])
                  earlier)
              earlier;
          ]
      in
      let n, dt = pick candidates in
      add n dt)
    choices;
  let nodes = Array.of_list (List.rev !nodes) in
  let n = Array.length nodes in
  {
    ins;
    nodes;
    types = !types;
    outs =
      Array.init
        (min (min nouts n) (P.max_operands - Array.length ins))
        (fun k -> n - 1 - k);
  }

let pp_drawn ppf d =
  Format.fprintf ppf "%d operands, %d nodes, outs %a" (Array.length d.ins)
    (Array.length d.nodes)
    (Format.pp_print_list Format.pp_print_int)
    (Array.to_list d.outs)

let drawn =
  let open Gen in
  with_pp pp_drawn
    (let+ ins = array ~size:(int_range 0 3) (of_list ~pp:pp_dtype base_dtypes)
     and+ choices = array ~size:(int_range 1 8) (int_range 0 1_000_000)
     and+ nouts = int_range 1 3 in
     build ins choices nouts)

(* What C reads of [d]'s program, rendered as the support reader renders
   it. *)
let render d =
  let code (D.Any dt) = D.code dt in
  let line nd =
    match nd with
    | P.In k -> strf "in %d\n" k
    | Coord k -> strf "coord %d\n" k
    | Const (dt, s) -> strf "const %d %s\n" (code dt) (hex s)
    | Op1 (k, dt, a) -> strf "op1 %s %d %d\n" (op1_name k) (code dt) a
    | Op2 (k, a, b) -> strf "op2 %s %d %d\n" (op2_name k) a b
    | Op3 (k, a, b, c) -> strf "op3 %s %d %d %d\n" (op3_name k) a b c
  in
  String.concat "" (Array.to_list (Array.map line d.nodes))
  ^ "ins"
  ^ String.concat "" (Array.to_list (Array.map (fun t -> strf " %d" (code t)) d.ins))
  ^ "\nouts"
  ^ String.concat "" (Array.to_list (Array.map (strf " %d") d.outs))
  ^ "\n"

let pp_node ppf = function
  | P.In k -> Format.fprintf ppf "In %d" k
  | Coord k -> Format.fprintf ppf "Coord %d" k
  | Const (d, s) -> Format.fprintf ppf "Const (%s, %s)" (name d) (hex s)
  | Op1 (k, d, a) -> Format.fprintf ppf "Op1 (%s, %s, %d)" (op1_name k) (name d) a
  | Op2 (k, a, b) -> Format.fprintf ppf "Op2 (%s, %d, %d)" (op2_name k) a b
  | Op3 (k, a, b, c) ->
      Format.fprintf ppf "Op3 (%s, %d, %d, %d)" (op3_name k) a b c

let node = Testable.make ~pp:pp_node ~equal:( = )

let program d =
  P.v ~ins:(Iarray.of_array d.ins) (Iarray.of_array d.nodes)
    ~outs:(Iarray.of_array d.outs)

let law_program d =
  Array.iter
    (fun n ->
      cover "an Op3" (match n with P.Op3 _ -> true | _ -> false);
      cover "an Op2" (match n with P.Op2 _ -> true | _ -> false);
      cover "an Op1" (match n with P.Op1 _ -> true | _ -> false))
    d.nodes;
  let p = program d in
  equal ~msg:"C reads" string (render d) (Nx_kernel_support.prog p);
  equal ~msg:"ins" (list string)
    (Array.to_list (Array.map name d.ins))
    (List.map name (Iarray.to_list p.ins));
  equal ~msg:"outs" (array int) d.outs (Iarray.to_array p.outs);
  equal ~msg:"nodes" (array node) d.nodes (Iarray.to_array p.nodes);
  Array.iteri
    (fun i t ->
      equal ~msg:(strf "dtype %d" i) string (name t) (name (P.dtype p i)))
    d.types;
  let q = program d in
  equal ~msg:"equal programs" bool true (q = p);
  equal ~msg:"equal hashes" int (Hashtbl.hash p) (Hashtbl.hash q)

(* One node over drawn operands, its references in and around the operands'
   range, and its program built by v from the operands' nodes. *)
let one_node =
  let open Gen in
  let f32 = D.Any D.Float32 in
  with_pp
    (fun ppf (ins, n) ->
      Format.fprintf ppf "%d operands, %a" (Array.length ins) pp_node n)
    (let* ins = array ~size:(int_range 0 3) (of_list ~pp:pp_dtype base_dtypes) in
     let k = Array.length ins in
     let ref_ = int_range (-1) (k + 1) in
     let+ n =
       one_of
         [
           map (fun j -> P.In j) ref_;
           map (fun a -> P.Coord a) (int_range 0 Nx_array.Layout.max_rank);
           constant ~pp:pp_node (P.Const (f32, P.bits D.Float32 1.5));
           map (fun j -> P.Op1 (Copy, (if k > 0 && j >= 0 && j < k then ins.(j) else f32), j)) ref_;
           map (fun (j, l) -> P.Op2 (Binary Add, j, l)) (pair ref_ ref_);
           map (fun (c, j, l) -> P.Op3 (Where, c, j, l)) (triple ref_ ref_ ref_);
         ]
     in
     (ins, n))

let law_of_node (ins, n) =
  let k = Array.length ins in
  let built f = match f () with p -> Ok p | exception Invalid_argument _ -> Error () in
  let by_v =
    built (fun () ->
        P.v ~ins:(Iarray.of_array ins)
          (Iarray.init (k + 1) (fun i -> if i < k then P.In i else n))
          ~outs:[| k |])
  in
  cover "built" (Result.is_ok by_v);
  cover "refused" (Result.is_error by_v);
  match (by_v, built (fun () -> P.of_node ~ins:(Iarray.of_array ins) n)) with
  | Ok p, Ok q -> equal ~msg:"v's program" bool true (p = q)
  | Error (), Error () -> ()
  | Ok _, Error () -> fail "of_node refused what v builds"
  | Error (), Ok _ -> fail "of_node built what v refuses"

let test_program_refuses () =
  let f32 = D.Any D.Float32 in
  let refuses ~msg ?(ins = [| f32 |]) nodes outs =
    raises_match ~msg Exn.invalid_arg (fun () ->
        P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
          ~outs:(Iarray.of_array outs))
  in
  refuses ~msg:"a forward reference" [| P.Op1 (Copy, f32, 1); In 0 |] [| 0 |];
  refuses ~msg:"a reference to itself" [| P.Op1 (Copy, f32, 0) |] [| 0 |];
  refuses ~msg:"an operand past ins" [| P.In 1 |] [| 0 |];
  refuses ~msg:"an axis past max_rank" [| P.Coord Nx_array.Layout.max_rank |] [| 0 |];
  refuses ~msg:"a kind off its domain" [| P.In 0; Op2 (Binary Idiv, 0, 0) |] [| 1 |];
  refuses ~msg:"two dtypes"
    ~ins:[| f32; D.Any D.Float64 |]
    [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
    [| 2 |];
  refuses ~msg:"bits of the wrong width" [| P.Const (f32, "\000\000") |] [| 0 |];
  refuses ~msg:"a bool of 2" [| P.Const (D.Any D.Bool, "\002") |] [| 0 |];
  refuses ~msg:"no output" [| P.In 0 |] [||];
  refuses ~msg:"an output past the nodes" [| P.In 0 |] [| 1 |];
  refuses ~msg:"seventeen operands and outputs"
    ~ins:(Array.make 16 f32)
    [| P.In 0 |] [| 0 |];
  ignore
    (P.v ~ins:(Iarray.init 15 (fun _ -> f32)) [| P.In 0 |] ~outs:[| 0 |])

let test_dtype_refuses () =
  let p = P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0 |] in
  raises_match ~msg:"past the nodes" Exn.invalid_arg (fun () -> P.dtype p 1);
  raises_match ~msg:"negative" Exn.invalid_arg (fun () -> P.dtype p (-1))

(* Representation: C reads a program by nx_prog.h's codes, so each
   constructor's tag, each field's position and each kind's code is the one
   the header names, and a reordered declaration fails here. *)

(* The header's name of the kind of arity [a] named [name]. *)
let header a name = strf "NX_OP%d_%s" a (String.uppercase_ascii name)

(* Field [off] past the one the header names [name] of [v], as C reads it. *)
let field ?(off = 0) v name =
  let r = Obj.repr v and i = Nx_kernel_support.code name + off in
  if i >= Obj.size r then
    failf "%s + %d is field %d of a block of %d" name off i (Obj.size r);
  Obj.field r i

let repr =
  Testable.make
    ~pp:(fun ppf o ->
      if Obj.is_int o then Format.fprintf ppf "%d" (Obj.obj o : int)
      else Format.fprintf ppf "a block of tag %d" (Obj.tag o))
    ~equal:( == )

let test_codes () =
  let code = Nx_kernel_support.code and tag = Nx_kernel_support.tag in
  List.iter
    (fun k ->
      equal ~msg:(op1_name k) int
        (code (header 1 (op1_name k)))
        (Nx_kernel_support.op1 k))
    op1s;
  List.iter
    (fun k ->
      equal ~msg:(op2_name k) int
        (code (header 2 (op2_name k)))
        (Nx_kernel_support.op2 k))
    op2s;
  List.iter
    (fun k ->
      equal ~msg:(op3_name k) int (code (header 3 (op3_name k))) (tag k))
    P.[ Where; Fma ];
  let f64 = D.Any D.Float64 and one = P.bits D.Float64 1. in
  let exp = P.Unary Exp and add = P.Binary Add in
  List.iter
    (fun (name, t) -> equal ~msg:name int (code name) t)
    [
      ("NX_NODE_IN", tag (P.In 3));
      ("NX_NODE_COORD", tag (P.Coord 3));
      ("NX_NODE_CONST", tag (P.Const (f64, one)));
      ("NX_NODE_OP1", tag (P.Op1 (exp, f64, 5)));
      ("NX_NODE_OP2", tag (P.Op2 (add, 6, 7)));
      ("NX_NODE_OP3", tag (P.Op3 (Where, 4, 6, 7)));
      ("NX_OP2_BINARY", tag add);
      ("NX_OP2_COMPARE", tag (P.Compare Less));
    ];
  let p = P.v ~ins:[| f64 |] [| P.In 0 |] ~outs:[| 0 |] in
  let op1 = P.Op1 (exp, f64, 5) and op2 = P.Op2 (add, 6, 7) in
  let op3 = P.Op3 (Where, 4, 6, 7) and cst = P.Const (f64, one) in
  List.iter
    (fun (name, read, want) -> equal ~msg:name repr want read)
    [
      ("NX_PROG_INS", field p "NX_PROG_INS", Obj.repr p.ins);
      ("NX_PROG_NODES", field p "NX_PROG_NODES", Obj.repr p.nodes);
      ("NX_PROG_OUTS", field p "NX_PROG_OUTS", Obj.repr p.outs);
      ("NX_IN_OPERAND", field (P.In 3) "NX_IN_OPERAND", Obj.repr 3);
      ("NX_COORD_AXIS", field (P.Coord 2) "NX_COORD_AXIS", Obj.repr 2);
      ("NX_CONST_DTYPE", field cst "NX_CONST_DTYPE", Obj.repr f64);
      ("NX_CONST_BITS", field cst "NX_CONST_BITS", Obj.repr one);
      ("NX_OP1_KIND", field op1 "NX_OP1_KIND", Obj.repr exp);
      ("NX_OP1_DTYPE", field op1 "NX_OP1_DTYPE", Obj.repr f64);
      ("NX_OP1_OPERAND", field op1 "NX_OP1_OPERAND", Obj.repr 5);
      ("NX_OP2_KIND", field op2 "NX_OP2_KIND", Obj.repr add);
      ("NX_OP2_OPERANDS", field op2 "NX_OP2_OPERANDS", Obj.repr 6);
      ("NX_OP2_OPERANDS + 1", field ~off:1 op2 "NX_OP2_OPERANDS", Obj.repr 7);
      ("NX_OP3_KIND", field op3 "NX_OP3_KIND", Obj.repr P.Where);
      ("NX_OP3_OPERANDS", field op3 "NX_OP3_OPERANDS", Obj.repr 4);
      ("NX_OP3_OPERANDS + 1", field ~off:1 op3 "NX_OP3_OPERANDS", Obj.repr 6);
      ("NX_OP3_OPERANDS + 2", field ~off:2 op3 "NX_OP3_OPERANDS", Obj.repr 7);
    ]

let floats =
  Gen.one_of
    [
      Gen.float;
      Gen.of_list ~pp:Format.pp_print_float
        [ 0.; -0.; 1.; -1.5; 448.; 1e30; -1e-40; Float.nan; infinity; neg_infinity ];
    ]

let tests =
  [
    group "domains"
      [
        test "every kind and dtype as prog.mli states" test_domains;
        test "each base dtype a kind takes is computed by nx_kinds.h"
          test_domains_computed;
        test "accepts2 allocates nothing" test_domains_allocate_nothing;
      ];
    group "bits"
      [
        prop "bits are a one-element array's bytes"
          (Gen.pair (Gen.of_list ~pp:pp_dtype D.all) floats)
          law_bits;
        test "refuses an int outside the dtype's range" test_bits_refuse;
      ];
    group "programs"
      [
        prop "a program holds what v was given, as C reads it" drawn
          law_program;
        test "v refuses ill-formed programs" test_program_refuses;
        prop "of_node is v over the operands' nodes" one_node law_of_node;
        test "dtype refuses a node past the program" test_dtype_refuses;
        test "each tag, field and kind is nx_prog.h's code" test_codes;
      ];
  ]

let () = exit (run "nx_kernel.prog" tests)
