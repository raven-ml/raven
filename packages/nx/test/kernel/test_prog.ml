(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Programs: the kinds' domains against the table prog.mli states, constant
   bits against an array's bytes, and programs against what they were made
   of. *)

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
      Cos; Tan; Asin; Acos; Atan; Sinh; Cosh; Tanh; Asinh; Acosh; Atanh; Erf;
      Floor; Ceil; Round; Trunc;
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

(* Each kind's name in messages. *)
let named ks names k = List.assoc k (List.combine ks names)

let unary_name =
  named unaries
    [
      "neg"; "recip"; "abs"; "sign"; "sqrt"; "exp"; "exp2"; "log"; "log2";
      "log1p"; "expm1"; "sin"; "cos"; "tan"; "asin"; "acos"; "atan"; "sinh";
      "cosh"; "tanh"; "asinh"; "acosh"; "atanh"; "erf"; "floor"; "ceil";
      "round"; "trunc";
    ]

let binary_name =
  named binaries
    [
      "add"; "sub"; "mul"; "fdiv"; "idiv"; "mod"; "pow"; "atan2"; "maximum";
      "minimum"; "and"; "or"; "xor"; "threefry";
    ]

let compare_name = named compares [ "equal"; "not_equal"; "less"; "less_equal" ]

let op1_name = function
  | P.Copy -> "copy"
  | Cast -> "cast"
  | Bitcast -> "bitcast"
  | Unary u -> unary_name u

let op2_name = function
  | P.Binary b -> binary_name b
  | Compare c -> compare_name c

let op3_name = function P.Where -> "where" | Fma -> "fma"

(* prog.mli's domains, by kind of number: "Neg, Recip, Sign, Add, Sub, Mul,
   Pow: every dtype but booleans. Fdiv: floats and complex. Idiv: integers.
   Abs, Floor, Ceil, Round, Trunc, Mod, Fma: floats and integers. Sqrt, Exp,
   Exp2, Log, Log2, and Sin to Tanh: floats and complex. Asinh, Acosh, Atanh:
   complex. Log1p, Expm1, Erf, Atan2: floats. And, Or, Xor: integers and
   booleans. Threefry: uint64. Maximum, Minimum and the comparisons: every
   dtype. Copy: every dtype, into itself. Cast: every pair. Bitcast: every
   pair of one width. Where: a boolean, then two operands of any one
   dtype." *)
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
  | Sqrt | Exp | Exp2 | Log | Log2 | Sin | Cos | Tan | Asin | Acos | Atan
  | Sinh | Cosh | Tanh ->
      [ Float; Complex ]
  | Log1p | Expm1 | Erf -> [ Float ]
  | Asinh | Acosh | Atanh -> [ Complex ]

let binary_domain : P.binary -> cls list = function
  | Add | Sub | Mul -> numbers
  | Fdiv -> [ Float; Complex ]
  | Idiv -> integers
  | Mod -> reals
  | Pow -> numbers
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
    outs = Array.init (min nouts n) (fun k -> n - 1 - k);
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

(* Whether [n] reads [k] operands, each once, in order. *)
let reads_in_order k (n : P.node) =
  match n with
  | Const _ | Coord _ -> k = 0
  | Op1 (_, _, 0) -> k = 1
  | Op2 (_, 0, 1) -> k = 2
  | Op3 (_, 0, 1, 2) -> k = 3
  | In _ | Op1 _ | Op2 _ | Op3 _ -> false

(* A kind of two and of three operands reading them in order, and one out
   of order, which one_node seldom draws. *)
let singles =
  let f32 = D.Any D.Float32 and b = D.Any D.Bool in
  [
    ([| f32; f32 |], P.Op2 (Compare Less, 0, 1));
    ([| b; f32; f32 |], P.Op3 (Where, 0, 1, 2));
    ([| f32; f32 |], P.Op2 (Binary Sub, 1, 0));
  ]

let law_single_of_node (ins, n) =
  match P.of_node ~ins:(Iarray.of_array ins) n with
  | exception Invalid_argument _ -> ()
  | p ->
      let want = if reads_in_order (Array.length ins) n then Some n else None in
      cover "one node" (Option.is_some want);
      cover "not one node" (Option.is_none want);
      cover "three operands"
        (Array.length ins = 3 && match n with Op3 (_, 0, 1, 2) -> true | _ -> false);
      equal (option node) want (P.single p);
      equal ~msg:"is_single" bool (Option.is_some want) (P.is_single p)

(* Programs of_node does not build: two outputs, two kind nodes. *)
let test_single () =
  let f32 = D.Any D.Float32 in
  let prog nodes outs =
    P.v ~ins:[| f32 |] (Iarray.of_array nodes) ~outs:(Iarray.of_array outs)
  in
  let none ~msg p =
    equal ~msg (option node) None (P.single p);
    equal ~msg bool false (P.is_single p)
  in
  none ~msg:"two outputs" (prog [| In 0; Op1 (Unary Neg, f32, 0) |] [| 1; 1 |]);
  none ~msg:"two nodes"
    (prog [| In 0; Op1 (Unary Neg, f32, 0); Op1 (Unary Neg, f32, 1) |] [| 2 |])

(* One operand, one output, and that output its operand's In node. *)
let law_identity d =
  let want =
    Array.length d.ins = 1 && Array.length d.outs = 1
    && d.nodes.(d.outs.(0)) = P.In 0
  in
  cover "the identity" want;
  equal bool want (P.is_identity (program d))

let f32 = D.Any D.Float32

let identity =
  { ins = [| f32 |]; nodes = [| P.In 0 |]; types = [| f32 |]; outs = [| 0 |] }

let identities =
  let two = [| f32; f32 |] in
  [
    ("the identity", identity, true);
    ( "Copy of the operand",
      {
        identity with
        nodes = [| In 0; Op1 (Copy, f32, 0) |];
        types = two;
        outs = [| 1 |];
      },
      false );
    ( "the operand twice",
      { identity with nodes = [| In 0; In 0 |]; types = two; outs = [| 1 |] },
      true );
    ("two outputs", { identity with outs = [| 0; 0 |] }, false);
    ("two operands", { identity with ins = two }, false);
    ( "a constant",
      {
        identity with
        ins = [||];
        nodes = [| Const (f32, P.bits D.Float32 1.) |];
      },
      false );
  ]

let test_identity (_, d, want) = equal bool want (P.is_identity (program d))

let pp_prog ppf (p : P.t) =
  let list pp ppf xs =
    let sep ppf () = Format.fprintf ppf "; " in
    Format.pp_print_list ~pp_sep:sep pp ppf (Iarray.to_list xs)
  in
  let dtype ppf d = Format.pp_print_string ppf (name d) in
  Format.fprintf ppf "{ ins = [%a]; nodes = [%a]; outs = [%a] }" (list dtype)
    p.ins (list pp_node) p.nodes (list Format.pp_print_int) p.outs

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
  refuses ~msg:"a bit of 2" [| P.Const (D.Any D.Bit, "\002") |] [| 0 |];
  refuses ~msg:"an int4 of 16" [| P.Const (D.Any D.Int4, "\016") |] [| 0 |];
  (* A boolean byte is true iff it is not zero: each is an element. *)
  ignore (P.v ~ins:[||] [| P.Const (D.Any D.Bool, "\002") |] ~outs:[| 0 |]);
  refuses ~msg:"no output" [| P.In 0 |] [||];
  refuses ~msg:"an output past the nodes" [| P.In 0 |] [| 1 |]

let test_dtype_refuses () =
  let p = P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0 |] in
  raises_match ~msg:"past the nodes" Exn.invalid_arg (fun () -> P.dtype p 1);
  raises_match ~msg:"negative" Exn.invalid_arg (fun () -> P.dtype p (-1))

(* Expressions *)

(* The element of [dt] whose bits are [b]. *)
let element (type v s) (dt : (v, s) D.t) b : v =
  A.get (A.v dt (Nx_array.Layout.contiguous [||]) (Rig.Buffer.of_string b)) [||]

(* An expression of some dtype. *)
type ex = Ex : ('v, 's) P.E.t -> ex

(* [y] at [x]'s dtype, which a drawn node gives them both. *)
let like (type v s w r) (x : (v, s) P.E.t) (y : (w, r) P.E.t) : (v, s) P.E.t =
  match D.equal_witness (P.E.dtype y) (P.E.dtype x) with
  | Some Equal -> y
  | None -> failwith "a drawn node's operands differ in dtype"

(* [d]'s nodes as expressions, each over its operands' expressions, so that a
   node read twice is one value. A leaf [n] is [e] where [leaf n] is
   [Some e]. *)
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

let any (Ex x) = P.E.any x
let exprs d = Array.map any (typed d)

(* of_exprs's nodes and outputs as prog.mli states them for [d]'s outputs:
   the nodes they reach, each structure once, in the order a depth-first
   walk from the outputs, operands left to right, finishes them. *)
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
      | Op2 (_, a, b) -> visit a; visit b
      | Op3 (_, a, b, c) -> visit a; visit b; visit c
    end
  in
  Array.iter visit outs;
  seen

(* [d] without the nodes its outputs do not reach, whose kinds a library
   may decline. *)
let pruned d =
  let keep = reach d d.outs in
  let at = Array.make (Array.length d.nodes) (-1) and count = ref 0 in
  Array.iteri (fun i k -> if k then (at.(i) <- !count; incr count)) keep;
  let renumber = function
    | (P.In _ | Coord _ | Const _) as nd -> nd
    | Op1 (k, dt, a) -> Op1 (k, dt, at.(a))
    | Op2 (k, a, b) -> Op2 (k, at.(a), at.(b))
    | Op3 (k, a, b, c) -> Op3 (k, at.(a), at.(b), at.(c))
  in
  let pick a = Array.of_list (List.filteri (fun i _ -> keep.(i)) (Array.to_list a)) in
  {
    d with
    nodes = Array.map renumber (pick d.nodes);
    types = pick d.types;
    outs = Array.map (fun o -> at.(o)) d.outs;
  }

(* Random elements of [dt]: any bits, a boolean's [0] or [1]. *)
let random_bits rand (D.Any dt) n =
  let byte () = Char.chr (Random.State.int rand 256) in
  let one () =
    if D.equal dt D.Bool then String.make 1 (Char.chr (Random.State.int rand 2))
    else String.init (max 1 (D.bits dt / 8)) (fun _ -> byte ())
  in
  Array.init n (fun _ -> one ())

let small = [| 2; 3; 2 |]

let random_elements rand d n =
  Elements.of_elements d small (random_bits rand d n)

(* [p]'s outputs over [loads] at [shape], [2; 3; 2] by default, computed by
   nx.cpu, as their elements' bits. *)
let compute ?(shape = small) (p : P.t) loads =
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

(* [p]'s outputs over [loads] as {!compute} gives them, with every NaN
   element, and every NaN part of a complex element, as the string "nan":
   only a move specifies a NaN result's bits, so two programs computing
   the same value may give NaNs of different signs or payloads. Every
   other element keeps its bits. *)
let any_nan (p : P.t) loads =
  let blind d e = if Elements.nan_of d e then "nan" else e in
  let one (D.Any dt as d) e =
    if not (D.is D.Complex dt) then blind d e
    else
      let h = String.length e / 2 in
      let part = if h = 4 then D.Any D.Float32 else D.Any D.Float64 in
      blind part (String.sub e 0 h) ^ blind part (String.sub e h h)
  in
  let out i es = Array.map (one (P.dtype p (Iarray.get p.outs i))) es in
  Result.map (Array.mapi out) (compute p loads)

(* Whether each element of each output of [d]'s program over [loads] reads a
   NaN's sign or payload: a Bitcast of a float reads its operand's bits, and
   where that operand is a NaN an earlier kind left them unspecified
   (prog.mli, Results), so two programs computing the same values may give
   different bits from there on, as nx.cpu's vector and scalar code may. A
   map computes each index from the operands' elements at it alone, so each
   element is followed apart. *)
let nan_bits d loads =
  let p = program (pruned d) in
  let nodes = Iarray.to_array p.nodes in
  let value i =
    let q =
      P.v ~ins:p.ins (Iarray.of_array (Array.sub nodes 0 (i + 1))) ~outs:[| i |]
    in
    match compute q loads with Ok a -> Some a.(0) | Error _ -> None
  in
  let n = Array.fold_left ( * ) 1 small in
  let read = Array.make (Array.length nodes) None in
  let rec reads i =
    match read.(i) with
    | Some r -> r
    | None ->
        let any = Array.make n false in
        let join j =
          Array.iteri (fun e b -> if b then any.(e) <- true) (reads j)
        in
        (match nodes.(i) with
        | P.In _ | Coord _ | Const _ -> ()
        | Op1 (k, _, x) -> (
            join x;
            let (D.Any dt) = P.dtype p x in
            match (k, value x) with
            | Bitcast, Some es when D.is D.Float dt || D.is D.Complex dt ->
                let d = D.Any dt in
                Array.iteri
                  (fun e el -> if Elements.nan_of d el then any.(e) <- true)
                  es
            | _ -> ())
        | Op2 (_, x, y) -> join x; join y
        | Op3 (_, x, y, z) -> join x; join y; join z);
        read.(i) <- Some any;
        any
  in
  Array.map reads (Iarray.to_array p.outs)

(* [r], {!any_nan}'s outputs, with each element [nan_bits] marks unspecified. *)
let blinded marks r =
  Result.map
    (Array.mapi (fun o es ->
         Array.mapi (fun e v -> if marks.(o).(e) then "NaN bits" else v) es))
    r

(* Substitution *)

let structural = Testable.structural ~pp:pp_prog

(* [p] as a drawn program. *)
let drawn_of (p : P.t) =
  {
    ins = Iarray.to_array p.ins;
    nodes = Iarray.to_array p.nodes;
    types = Array.init (Iarray.length p.nodes) (P.dtype p);
    outs = Iarray.to_array p.outs;
  }

(* [subst ?outs ~ins f p] of [d]'s program [p], checked against prog.mli:
   of_exprs's program over [ins] of the outputs [outs] selects, each leaf [n]
   they reach replaced by [f n]'s expression where it is [Some], and as v
   builds it; [f] called once on each such leaf and on no other node. Is the
   result. *)
let check_subst ?outs ~ins f d =
  let ins = Iarray.of_array ins in
  let calls = ref [] in
  let called n =
    calls := n :: !calls;
    Option.map any (f n)
  in
  let s = P.subst ?outs ~ins called (program d) in
  let picked =
    match outs with
    | None -> d.outs
    | Some o -> Array.map (Array.get d.outs) (Iarray.to_array o)
  in
  let reached = reach d picked in
  let leaves =
    List.filteri
      (fun i (nd : P.node) ->
        reached.(i)
        && match nd with In _ | Coord _ | Const _ -> true | _ -> false)
      (Array.to_list d.nodes)
  in
  equal ~msg:"f's calls" (list node)
    (List.sort compare leaves)
    (List.sort compare !calls);
  let e = typed ~leaf:f d in
  let want = P.of_exprs ~ins (Iarray.of_array (Array.map (fun o -> any e.(o)) picked)) in
  equal ~msg:"of_exprs's program" structural want s;
  equal ~msg:"as v builds it" bool true
    (s = P.v ~ins:s.ins s.nodes ~outs:s.outs);
  s

let law_subst_keeps d =
  let s = check_subst ~ins:d.ins (fun _ -> None) d in
  let nodes, outs = lowered d in
  equal ~msg:"nodes" (array node) nodes (Iarray.to_array s.nodes);
  equal ~msg:"outs" (array int) outs (Iarray.to_array s.outs);
  cover "fewer nodes than drawn"
    (Iarray.length s.nodes < Array.length d.nodes)

let without k a =
  Array.of_list (List.filteri (fun i _ -> i <> k) (Array.to_list a))

(* The substitution that binds operand [k] of [ins] to [x]: [In k] is [x],
   and [In i] for [i > k] is [In (i - 1)]. *)
let binds ins k x : P.node -> ex option = function
  | P.In i when i = k -> Some x
  | In i when i > k ->
      let (D.Any dt) = ins.(i) in
      Some (Ex (P.E.input (i - 1) dt))
  | In _ | Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _ -> None

(* An expression drawn over [ins], its last node cast to [dt]. *)
let ending ins choices dt =
  let d = build ins choices 1 in
  let last = Array.length d.nodes - 1 in
  let d = { d with nodes = Array.append d.nodes [| P.Op1 (Cast, dt, last) |] } in
  (typed d).(last + 1)

let binding =
  let open Gen in
  with_pp
    (fun ppf (d, k, _, _) -> Format.fprintf ppf "%a, operand %d" pp_drawn d k)
    (let+ d = drawn
     and+ k = int_range 0 2
     and+ choices = array ~size:(int_range 1 6) (int_range 0 1_000_000)
     and+ seed = int in
     (d, k, choices, seed))

(* [d]'s operand [k] bound to [x] by subst, checked as check_subst does over
   [d]'s other operands, and its outputs' bits on nx.cpu against [d]'s over
   [x]'s value, over operands drawn from [seed]. Whether nx.cpu computed
   [x]. *)
let check_bind d k x seed =
  let others = without k d.ins in
  let s = check_subst ~ins:others (binds d.ins k x) d in
  equal ~msg:"ins" (list string)
    (Array.to_list (Array.map name others))
    (List.map name (Iarray.to_list s.ins));
  let q = P.of_exprs ~ins:(Iarray.of_array others) [| any x |] in
  let rand = Random.State.make [| seed |] in
  let loads = Array.map (fun t -> random_elements rand t 12) others in
  match compute q loads with
  | Error _ -> false
  | Ok out ->
      let given = Elements.of_elements d.ins.(k) small out.(0) in
      let all =
        Array.init (Array.length d.ins) (fun i ->
            if i = k then given else loads.(if i < k then i else i - 1))
      in
      let marks = nan_bits d all in
      equal ~msg:"outputs' bits" outputs
        (blinded marks (any_nan (program (pruned d)) all))
        (blinded marks (any_nan s loads));
      true

let law_bind (d, k, choices, seed) =
  assume (k < Array.length d.ins);
  cover "p reads operand k" (Array.mem (P.In k) d.nodes);
  cover "an operand after k" (k < Array.length d.ins - 1);
  cover "k is 0" (k = 0);
  cover "x computed"
    (check_bind d k (ending (without k d.ins) choices d.ins.(k)) seed)

(* Expressions whose value is an operand, a coordinate or a constant, bound
   at k = 0, with p reading or outputting In k. *)
let test_subst_leaves () =
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 and i64 = D.Any D.Int64 in
  let p ins nodes outs =
    drawn_of
      (P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
         ~outs:(Iarray.of_array outs))
  in
  let sum = p [| f32; f32 |] [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |] [| 2 |] in
  equal ~msg:"an operand" bool true
    (check_bind sum 0 (Ex (P.E.input 0 D.Float32)) 1);
  let scaled =
    p [| i64; f32 |]
      [| P.In 0; In 1; Op1 (Cast, f32, 0); Op2 (Binary Mul, 2, 1) |]
      [| 3 |]
  in
  equal ~msg:"a coordinate" bool true (check_bind scaled 0 (Ex (P.E.coord 1)) 2);
  let passed = p [| f32; f64 |] [| P.In 0; In 1 |] [| 0; 1 |] in
  equal ~msg:"a constant, output as p's In k" bool true
    (check_bind passed 0 (Ex (P.E.const D.Float32 2.5)) 3)

(* A drawn program with [Coord a] and its double added, the double an output,
   so that a kind reads a coordinate; each axis's offset. *)
let shifting =
  let open Gen in
  with_pp
    (fun ppf (d, off) ->
      Format.fprintf ppf "%a, offsets %a" pp_drawn d Elements.pp_ints off)
    (let+ d = drawn
     and+ a = int_range 0 2
     and+ off = array ~size:(int_range 3 3) (int_range 0 3) in
     let n = Array.length d.nodes in
     let added = [| P.Coord a; Op2 (Binary Add, n, n) |] in
     let i64 = D.Any D.Int64 in
     ( {
         d with
         nodes = Array.append d.nodes added;
         types = Array.append d.types [| i64; i64 |];
         outs = Array.append d.outs [| n + 1 |];
       },
       off ))

(* subst adding [off] to each coordinate. *)
let law_subst_coords (d, off) =
  let shift : P.node -> ex option = function
    | Coord c when off.(c) <> 0 ->
        Some (Ex P.E.(coord c + const D.Int64 (Int64.of_int off.(c))))
    | _ -> None
  in
  ignore (check_subst ~ins:d.ins shift d);
  let reached = (pruned d).nodes in
  cover "a coordinate shifted"
    (Array.exists (function P.Coord c -> off.(c) <> 0 | _ -> false) reached);
  cover "a coordinate kept"
    (Array.exists (function P.Coord c -> off.(c) = 0 | _ -> false) reached)

(* A drawn program, outputs selected among its own by index. *)
let selecting =
  let open Gen in
  with_pp
    (fun ppf (d, sel) ->
      Format.fprintf ppf "%a, selecting %a" pp_drawn d Elements.pp_ints sel)
    (let+ d = drawn
     and+ sel = array ~size:(int_range 1 4) (int_range 0 1_000) in
     (d, Array.map (fun k -> k mod Array.length d.outs) sel))

(* subst's outputs with [outs] are [d]'s at [outs], in that order. *)
let law_subst_outs (d, sel) =
  ignore (check_subst ~outs:(Iarray.of_array sel) ~ins:d.ins (fun _ -> None) d);
  let n = Array.length sel in
  cover "an output twice"
    (List.length (List.sort_uniq Int.compare (Array.to_list sel)) < n);
  cover "an output dropped"
    (List.exists (fun o -> not (Array.mem o sel)) (List.init (Array.length d.outs) Fun.id));
  cover "outputs reordered"
    (List.exists (fun j -> sel.(j) > sel.(j + 1)) (List.init (n - 1) Fun.id))

let test_subst_refuses () =
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 in
  let p =
    P.of_exprs ~ins:[| f32; f64 |]
      P.E.[| any (input 0 D.Float32); any (input 1 D.Float64); any (coord 0) |]
  in
  let refuses ~msg ?outs ins f =
    raises_match ~msg Exn.invalid_arg (fun () ->
        P.subst ?outs ~ins:(Iarray.of_array ins) f p)
  in
  let in1 x : P.node -> P.E.any option = function
    | P.In 1 -> Some x
    | _ -> None
  in
  let keep _ = None in
  refuses ~msg:"an expression of another dtype" [| f32 |]
    (in1 P.E.(any (const D.Float32 1.)));
  refuses ~msg:"a coordinate's expression of another dtype" [| f32; f64 |]
    (function P.Coord _ -> Some P.E.(any (const D.Int32 1l)) | _ -> None);
  refuses ~msg:"a read of an operand the result lacks" [| f32 |]
    (in1 P.E.(any (cast D.Float64 (input 1 D.Float32))));
  refuses ~msg:"a read of an operand at another dtype" [| f32 |]
    (in1 P.E.(any (input 0 D.Float64)));
  refuses ~msg:"a kept operand past ins" [| f32 |] keep;
  refuses ~msg:"a kept operand at another dtype" [| f32; f32 |] keep;
  refuses ~msg:"an output past p's" ~outs:[| 3 |] [| f32; f64 |] keep;
  refuses ~msg:"a negative output" ~outs:[| -1 |] [| f32; f64 |] keep;
  refuses ~msg:"no output" ~outs:[||] [| f32; f64 |] keep;
  ignore (P.subst ~ins:[| f32 |] (in1 P.E.(any (const D.Float64 1.))) p);
  (* A leaf the selected outputs do not reach is not substituted, and need
     not fit [ins]. *)
  let unreached : P.node -> P.E.any option = function
    | P.In 1 -> failwith "subst substituted an unreached leaf"
    | _ -> None
  in
  ignore (P.subst ~outs:[| 0; 2 |] ~ins:[| f32 |] unreached p)

(* of_exprs of [d]'s outputs, the odd ones from a second build, so that every
   node they reach is a value apart from the first build's and merges by its
   structure; checked against {!lowered} and against [d]'s program over
   operands drawn from [seed]. Is the lowered program. *)
let check_exprs d seed =
  let first = exprs d and second = exprs d in
  let out k i = if k mod 2 = 0 then first.(i) else second.(i) in
  let q = P.of_exprs ~ins:(Iarray.of_array d.ins) (Iarray.of_array (Array.mapi out d.outs)) in
  let nodes, outs = lowered d in
  equal ~msg:"nodes" (array node) nodes (Iarray.to_array q.nodes);
  equal ~msg:"outs" (array int) outs (Iarray.to_array q.outs);
  equal ~msg:"ins" (list string) (Array.to_list (Array.map name d.ins))
    (List.map name (Iarray.to_list q.ins));
  let rand = Random.State.make [| seed |] in
  let loads = Array.map (fun t -> random_elements rand t 12) d.ins in
  let marks = nan_bits d loads in
  let want = blinded marks (any_nan (program (pruned d)) loads) in
  equal ~msg:"outputs' bits" outputs want (blinded marks (any_nan q loads));
  q

let law_of_exprs (d, seed) =
  let q = check_exprs d seed in
  cover "outputs of two builds" (Array.length d.outs >= 2);
  cover "fewer nodes than drawn" (Iarray.length q.nodes < Array.length d.nodes)

(* Programs grown node by node, up to 400, so that the walk's tables grow,
   a third of them up to 80, so that lowerings that scan and lowerings that
   cross to tables are both reached: each node is an operand, an axis, a
   constant, a copy of an earlier node's structure, or a kind over earlier
   nodes of the dtypes it takes. Constants
   are small integers, [-0.], NaNs with payloads, and the same bits in dtypes
   of one width. Outputs repeat and leave nodes unreached. *)

(* Bits that some dtypes of each width share: [-0.], three NaN payloads and
   [1.] at float32 and float64, also an int32, int64 or uint64. *)
let shared_bits =
  let four v = let b = Bytes.create 4 in Bytes.set_int32_ne b 0 v; Bytes.to_string b in
  let eight v = let b = Bytes.create 8 in Bytes.set_int64_ne b 0 v; Bytes.to_string b in
  List.map four [ 0x8000_0000l; 0x7fc0_0001l; 0x7fc0_0002l; 0xffc0_0001l; 0x3f80_0000l ]
  @ List.map eight
      [
        0x8000_0000_0000_0000L; 0x7ff8_0000_0000_0001L; 0x7ff8_0000_0000_0002L;
        0xfff8_0000_0000_0001L; 0x3ff0_0000_0000_0000L;
      ]

(* The constants of [dt] a grown program draws: [-0.] and the integers 0 to
   6, and the shared bits of [dt]'s width that its elements hold. *)
let constants (D.Any dt as d) =
  let ints = List.map (fun x -> P.bits dt (D.of_float dt x)) [ -0.; 0.; 1.; 2.; 3.; 4.; 5.; 6. ] in
  let shared =
    List.filter
      (fun b ->
        String.length b = D.bytes dt 1 && P.bits dt (element dt b) = b)
      shared_bits
  in
  List.map (fun b -> P.Const (d, b)) (ints @ shared)

let grow rand ins n nouts =
  let types = Array.make n (D.Any D.Int64) and nodes = Array.make n (P.Coord 0) in
  let int k = Random.State.int rand k in
  let pick l = List.nth l (int (List.length l)) in
  (* Earlier nodes of [i]'s dtype, [i] among them. *)
  let alike i k =
    List.filter (fun j -> same types.(j) types.(i)) (List.init k Fun.id)
  in
  (* Half the constants are shared bits, in a dtype of their width. *)
  let constant () =
    match int 2 with
    | 0 ->
        let d = pick base_dtypes in
        (pick (constants d), d)
    | _ ->
        let b = pick shared_bits in
        let holds (D.Any dt) =
          String.length b = D.bytes dt 1 && P.bits dt (element dt b) = b
        in
        let d = pick (List.filter holds base_dtypes) in
        (P.Const (d, b), d)
  in
  let leaf () =
    match int 3 with
    | 0 when Array.length ins > 0 ->
        let k = int (Array.length ins) in
        (P.In k, ins.(k))
    | 1 -> (P.Coord (int 3), D.Any D.Int64)
    | _ -> constant ()
  in
  let copy i = (P.Op1 (Copy, types.(i), i), types.(i)) in
  (* A kind over [i] and earlier nodes of the dtypes it takes. *)
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
            let d = match o with Compare _ -> D.Any D.Bool | Binary _ -> types.(i) in
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
  (* Each kind's first operand is the last kind made, so that the last node
     reaches most of the program. *)
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
  { ins; nodes; types; outs }

let grown =
  let open Gen in
  with_pp pp_drawn
    (let+ ins = array ~size:(int_range 0 3) (of_list ~pp:pp_dtype base_dtypes)
     and+ n = frequency [ (1, int_range 1 80); (2, int_range 1 400) ]
     and+ nouts = int_range 1 4
     and+ seed = int in
     grow (Random.State.make [| seed |]) ins n nouts)

(* Whether two of [q]'s constants have the same bits and two dtypes. *)
let shared_constant (q : P.t) =
  let consts =
    List.filter_map
      (function P.Const (d, b) -> Some (d, b) | _ -> None)
      (Iarray.to_list q.nodes)
  in
  List.exists
    (fun (d, b) -> List.exists (fun (d', b') -> b = b' && not (same d d')) consts)
    consts

let law_of_grown_exprs (d, seed) =
  let q = check_exprs d seed in
  let n = Iarray.length q.nodes and r = Array.length (pruned d).nodes in
  let nodes = Iarray.to_list q.nodes in
  let has f = List.exists f nodes in
  let is_nan (D.Any dt) b =
    match D.kind dt with Float -> Float.is_nan (element dt b) | _ -> false
  in
  let is_minus_zero (D.Any dt) b =
    match D.kind dt with
    | Float ->
        let v = element dt b in
        v = 0. && Float.sign_bit v
    | _ -> false
  in
  cover "at most 40 nodes" (n <= 40);
  cover "41 to 80 nodes" (n > 40 && n <= 80);
  cover "over 128 nodes" (n > 128);
  cover "a merge" (n < r);
  cover "a node unreached" (r < Array.length d.nodes);
  let rec repeats = function [] -> false | o :: os -> List.mem o os || repeats os in
  cover "an output repeated" (repeats (Array.to_list d.outs));
  cover "a -0. constant" (has (function P.Const (d, b) -> is_minus_zero d b | _ -> false));
  cover "a NaN constant" (has (function P.Const (d, b) -> is_nan d b | _ -> false));
  cover "one constant's bits in two dtypes" (shared_constant q);
  cover "a Where over two nodes"
    (has (function P.Op3 (Where, _, b, c) -> b <> c | _ -> false));
  cover "a Where over one node" (has (function P.Op3 (Where, _, b, c) -> b = c | _ -> false));
  cover "an Fma over three nodes"
    (has (function P.Op3 (Fma, a, b, c) -> a <> b && b <> c && a <> c | _ -> false));
  cover "an Fma over one node" (has (function P.Op3 (Fma, a, b, c) -> a = b && b = c | _ -> false))

let test_of_exprs_two_builds () =
  (* Two builds of one chain of 15000 nodes, each node a constant or a sum
     over the last: apart as values, one program, both outputs its last
     node. *)
  let chain () =
    let y = ref (P.E.input 0 D.Int64) in
    for i = 1 to 7499 do
      y := P.E.(!y + const D.Int64 (Int64.of_int i))
    done;
    P.E.any !y
  in
  let q = P.of_exprs ~ins:[| D.Any D.Int64 |] [| chain (); chain () |] in
  equal ~msg:"nodes" int 14999 (Iarray.length q.nodes);
  equal ~msg:"outs" (array int) [| 14998; 14998 |] (Iarray.to_array q.outs)

let test_of_exprs_fresh_copies () =
  (* A sum of two fresh builds of one sum, 16 levels deep: 2^17 - 1
     expressions, apart as values, of 17 nodes. A lowering that scanned
     its walked expressions while it held few nodes would compare billions
     of pairs. *)
  let rec tree d =
    if d = 0 then P.E.input 0 D.Int64
    else
      let a = tree (d - 1) in
      let b = tree (d - 1) in
      P.E.(a + b)
  in
  let q = P.of_exprs ~ins:[| D.Any D.Int64 |] [| P.E.any (tree 16) |] in
  let want =
    Array.init 17 (fun i : P.node ->
        if i = 0 then In 0 else Op2 (Binary Add, i - 1, i - 1))
  in
  equal ~msg:"nodes" (array node) want (Iarray.to_array q.nodes);
  equal ~msg:"outs" (array int) [| 16 |] (Iarray.to_array q.outs)

(* Each kind function makes the node its kind's constructor makes. *)
let test_exprs_kind_functions () =
  let ins = [| D.Any D.Float32; D.Any D.Float32; D.Any D.Bool; D.Any D.Uint64 |] in
  let x = P.E.input 0 D.Float32 and y = P.E.input 1 D.Float32 in
  let c = P.E.input 2 D.Bool and w = P.E.input 3 D.Uint64 in
  let lowered e = (P.of_exprs ~ins:(Iarray.of_array ins) [| e |]).nodes in
  let same name e e' =
    equal ~msg:name (array node) (Iarray.to_array (lowered e'))
      (Iarray.to_array (lowered e))
  in
  let u k f = same (unary_name k) (P.E.any (f x)) (P.E.any (P.E.op1 (Unary k) D.Float32 x)) in
  let b k f = same (binary_name k) (P.E.any (f x y)) (P.E.any (P.E.binary k x y)) in
  let cmp k f = same (compare_name k) (P.E.any (f x y)) (P.E.any (P.E.compare k x y)) in
  let complex_only = P.[ Asinh; Acosh; Atanh ] in
  List.iter2 u
    (List.filter (fun k -> not (List.mem k complex_only)) unaries)
    P.E.
      [
        neg; recip; abs; sign; sqrt; exp; exp2; log; log2; log1p; expm1; sin; cos;
        tan; asin; acos; atan; sinh; cosh; tanh; erf; floor; ceil; round; trunc;
      ];
  let z = P.E.input 0 D.Complex64 in
  let lowered_z e = (P.of_exprs ~ins:[| D.Any D.Complex64 |] [| e |]).nodes in
  List.iter2
    (fun k f ->
      equal ~msg:(unary_name k) (array node)
        (Iarray.to_array (lowered_z (P.E.any (P.E.op1 (Unary k) D.Complex64 z))))
        (Iarray.to_array (lowered_z (P.E.any (f z)))))
    complex_only P.E.[ asinh; acosh; atanh ];
  List.iter2 b
    P.[ Add; Sub; Mul; Fdiv; Mod; Pow; Atan2; Maximum; Minimum ]
    P.E.[ add; sub; mul; fdiv; rem; pow; atan2; maximum; minimum ];
  List.iter2 cmp compares P.E.[ equal; not_equal; less; less_equal ];
  same "threefry" (P.E.any (P.E.threefry w w)) (P.E.any (P.E.binary Threefry w w));
  same "logxor" (P.E.any (P.E.logxor w w)) (P.E.any (P.E.binary Xor w w));
  same "copy" (P.E.any (P.E.copy x)) (P.E.any (P.E.op1 Copy D.Float32 x));
  same "cast" (P.E.any (P.E.cast D.Int8 x)) (P.E.any (P.E.op1 Cast D.Int8 x));
  same "cast into its own dtype" (P.E.any (P.E.cast D.Float32 x)) (P.E.any x);
  same "bitcast" (P.E.any (P.E.bitcast D.Int32 x)) (P.E.any (P.E.op1 Bitcast D.Int32 x));
  same "where" (P.E.any (P.E.where c x y)) (P.E.any (P.E.op3 Where c x y));
  same "fma" (P.E.any (P.E.fma x y x)) (P.E.any (P.E.op3 Fma x y x))

(* Each operator over two operands lowers as the function prog.mli names,
   [>] and [>=] with their operands swapped. *)
let operators =
  let x = P.E.input 0 D.Float32 and y = P.E.input 1 D.Float32 in
  let c = P.E.input 0 D.Bool and e = P.E.input 1 D.Bool in
  let v = P.E.input 0 D.Uint64 and w = P.E.input 1 D.Uint64 in
  let f32 = D.Any D.Float32 and b = D.Any D.Bool and u64 = D.Any D.Uint64 in
  P.E.
    [
      ("~-", f32, any (~-x), any (neg x));
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
      ("&&", b, any (c && e), any (logand c e));
      ("||", b, any (c || e), any (logor c e));
      ("=", f32, any (x = y), any (equal x y));
      ("<>", f32, any (x <> y), any (not_equal x y));
      ("<", f32, any (x < y), any (less x y));
      ("<=", f32, any (x <= y), any (less_equal x y));
      (">", f32, any (x > y), any (less y x));
      (">=", f32, any (x >= y), any (less_equal y x));
    ]

let test_operator (_, dt, e, want) =
  let of_expr e = P.of_exprs ~ins:[| dt; dt |] [| e |] in
  equal structural (of_expr want) (of_expr e)

let test_exprs_refuse () =
  let f32 = D.Any D.Float32 and i32 = D.Any D.Int32 in
  let refuses ~msg want f =
    let got = match f () with _ -> false | exception Invalid_argument _ -> true in
    equal ~msg bool want got
  in
  List.iter
    (fun (D.Any a) ->
      let x = P.E.input 0 a in
      List.iter
        (fun (D.Any y) ->
          List.iter
            (fun k ->
              refuses ~msg:(strf "op1 %s %s %s" (op1_name k) (D.name a) (D.name y))
                (not (P.accepts1 k a y))
                (fun () -> P.E.op1 k y x))
            op1s)
        base_dtypes;
      List.iter
        (fun k ->
          refuses ~msg:(strf "binary %s %s" (binary_name k) (D.name a))
            (not (P.accepts2 (Binary k) a))
            (fun () -> P.E.binary k x x))
        binaries;
      List.iter
        (fun (D.Any c) ->
          let c = P.E.input 1 c in
          List.iter
            (fun k ->
              refuses
                ~msg:(strf "op3 %s %s %s" (op3_name k) (D.name (P.E.dtype c)) (D.name a))
                (not (P.accepts3 k (P.E.dtype c) a))
                (fun () -> P.E.op3 k c x x))
            P.[ Where; Fma ])
        base_dtypes)
    base_dtypes;
  (* The kinds of floats and complex numbers check their operand, as no type
     states that domain. *)
  let z = P.E.input 0 D.Complex64 and n = P.E.input 0 D.Int32 in
  let takes name (f : 'v 's. ('v, 's) P.E.t -> ('v, 's) P.E.t) =
    refuses ~msg:(name ^ " complex64") false (fun () -> f z);
    refuses ~msg:(name ^ " int32") true (fun () -> f n)
  in
  takes "sqrt" P.E.sqrt;
  takes "exp" P.E.exp;
  takes "exp2" P.E.exp2;
  takes "log" P.E.log;
  takes "log2" P.E.log2;
  takes "sin" P.E.sin;
  takes "cos" P.E.cos;
  takes "tan" P.E.tan;
  takes "asin" P.E.asin;
  takes "acos" P.E.acos;
  takes "atan" P.E.atan;
  takes "sinh" P.E.sinh;
  takes "cosh" P.E.cosh;
  takes "tanh" P.E.tanh;
  takes "asinh" P.E.asinh;
  takes "acosh" P.E.acosh;
  takes "atanh" P.E.atanh;
  raises_match ~msg:"an axis past max_rank" Exn.invalid_arg (fun () ->
      P.E.coord Nx_array.Layout.max_rank);
  raises_match ~msg:"a negative axis" Exn.invalid_arg (fun () -> P.E.coord (-1));
  raises_match ~msg:"a negative operand" Exn.invalid_arg (fun () ->
      P.E.input (-1) D.Float32);
  let x = P.E.any (P.E.input 0 D.Float32) in
  raises_match ~msg:"no output" Exn.invalid_arg (fun () ->
      P.of_exprs ~ins:[| f32 |] [||]);
  raises_match ~msg:"an operand past ins" Exn.invalid_arg (fun () ->
      P.of_exprs ~ins:[||] [| x |]);
  raises_match ~msg:"an operand of another dtype" Exn.invalid_arg (fun () ->
      P.of_exprs ~ins:[| i32 |] [| x |])

(* No bound on a program's operands and outputs: twenty operands, each an
   output, an expression of operand 20, and a node over operand 19. *)
let test_many_operands () =
  let f32 = D.Any D.Float32 in
  let ins n = Iarray.init n (fun _ -> f32) in
  let p =
    P.v ~ins:(ins 20) (Iarray.init 20 (fun k -> P.In k))
      ~outs:(Iarray.init 20 Fun.id)
  in
  equal ~msg:"v" (array int) (Array.init 20 Fun.id) (Iarray.to_array p.outs);
  let q = P.of_exprs ~ins:(ins 21) [| P.E.(any (input 20 D.Float32)) |] in
  equal ~msg:"of_exprs" (array node) [| P.In 20 |] (Iarray.to_array q.nodes);
  let r = P.of_node ~ins:(ins 20) (Op2 (Binary Add, 0, 19)) in
  equal ~msg:"of_node" node (Op2 (Binary Add, 0, 19))
    (Iarray.get r.nodes 20)

let floats =
  Gen.one_of
    [
      Gen.float;
      Gen.of_list ~pp:Format.pp_print_float
        [ 0.; -0.; 1.; -1.5; 448.; 1e30; -1e-40; Float.nan; infinity; neg_infinity ];
    ]

(* Ranges *)

(* An integer or boolean element as its value in an int64, unsigned ones
   zero-extended and compared unsigned, booleans 0 or 1. *)
let unsigned (D.Any dt) = D.is D.Unsigned dt

let value (D.Any dt) e =
  match (D.kind dt, D.bits dt) with
  | Boolean, _ -> if e.[0] = '\000' then 0L else 1L
  | Signed, 4 -> Int64.of_int (((Char.code e.[0] land 15) lxor 8) - 8)
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
      Any Int8; Any Uint8; Any Int16; Any Uint16; Any Int32; Any Uint32;
      Any Int64; Any Uint64; Any Int4; Any Uint4;
    ]

let i64 = D.Any D.Int64
let boolean = D.Any D.Bool
let bit = D.Any D.Bit
let booleans = [ boolean; bit ]
let is_bool (D.Any dt) = D.is D.Boolean dt

(* A value of [d]: small, at or beside an end, or anywhere; a bool's byte
   any byte, true where it is not 0. *)
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

(* A pair program's tree, each node of the dtype its maker gives it, over
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

(* A leaf of [d]: a coordinate, a constant, or a load, each cast to [d]
   where it is of another dtype. *)
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
          let* s = of_list ~pp:pp_dtype (flipped d :: (if byte then [ boolean ] else [])) in
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

let ins c = Array.append [| f32 |] c.loads

(* The program of every integer and boolean node of [c] as an output. *)
let all_ranged c =
  let p =
    P.v ~ins:(Iarray.of_array (ins c)) (Iarray.of_array c.nodes)
      ~outs:[| 0 |]
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
let offsets =
  Gen.of_list
    [ 0L; 0L; 5L; 125L; 254L; 32765L; 65533L; 0x7FFF_FFFDL; 0xFFFF_FFFDL;
      0x7FFF_FFFF_FFFF_FFF0L ]

let ranged_cases =
  let open Gen in
  with_pp pp_ranged
    (let* loads =
       array ~size:(int_range 1 3) (of_list ~pp:pp_dtype (booleans @ ints))
     in
     let* out = of_list ~pp:pp_dtype (booleans @ booleans @ ints) in
     let* t = tree loads 3 out in
     let* shape = array ~size:(constant 3) (int_range 1 4) in
     let* offsets = array ~size:(constant 3) offsets in
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
     {
       loads;
       nodes = lower t;
       shape;
       offsets;
       values = Array.of_list values;
     })


let elements1 d x = Elements.of_elements d [| 1 |] [| element_of d x |]

(* Every integer and boolean node of a drawn program lies, at every index of
   the box, within its range over the box: the least and greatest of each
   load and coordinate there. *)
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
  let shift : P.node -> ex option = function
    | Coord a when c.offsets.(a) <> 0L ->
        Some (Ex P.E.(coord a + const D.Int64 c.offsets.(a)))
    | _ -> None
  in
  let shifted = P.subst ~ins:p.ins (fun n -> Option.map any (shift n)) p in
  let values =
    match compute ~shape:c.shape shifted loads with
    | Ok v -> v
    | Error a -> failf "the program refused: %a" Nx_array_support.pp_answer a
  in
  let bounds =
    Array.concat
      [
        [| elements1 f32 0L; elements1 f32 0L |];
        Array.concat
          (List.mapi
             (fun k d ->
               let v = c.values.(k) in
               let pick f = Array.fold_left (fun a x -> if f (order d x a) then x else a) v.(0) v in
               let least = pick (fun o -> o < 0) and most = pick (fun o -> o > 0) in
               [| elements1 d least; elements1 d most |])
             (Array.to_list c.loads));
        Array.concat
          (List.init 3 (fun a ->
               let lo = c.offsets.(a) in
               let hi = Int64.add lo (Int64.of_int (c.shape.(a) - 1)) in
               [| elements1 i64 lo; elements1 i64 hi |]));
      ]
  in
  let q = P.ranges ~axes:3 p in
  (* It holds as v states. *)
  ignore (P.v ~ins:q.ins q.nodes ~outs:q.outs : P.t);
  let r =
    match compute ~shape:[| 1 |] q bounds with
    | Ok r -> r
    | Error a -> failf "the ranges refused: %a" Nx_array_support.pp_answer a
  in
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
    outs;
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
    | Op1 (Cast, d, x)
      when ranged x && not (is_bool (P.dtype p x) || is_bool d) ->
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
  cover "an integer or boolean computed from a float" (exists from_float outs)

(* A cast into bit over a coordinate's bounds [lo, hi]: [Cast Bit] of the
   coordinate, and of it with [true]. The range is that of [x <> 0] over
   [lo, hi], of bit's dtype. *)
let bit_casts =
  [ ("zero", (0L, 0L), ("\000", "\000")); ("apart from zero", (1L, 3L), ("\001", "\001"));
    ("across zero", (0L, 2L), ("\000", "\001")) ]

let test_bit_cast (_, (lo, hi), (want_lo, want_hi)) =
  let bit = D.Any D.Bit in
  let p =
    P.v ~ins:[||]
      [| P.Coord 0; Op1 (Cast, bit, 0); Const (bit, P.bits D.Bit true);
         Op2 (Binary And, 1, 2) |]
      ~outs:[| 3 |]
  in
  let q = P.ranges ~axes:1 p in
  let q = P.v ~ins:q.ins q.nodes ~outs:q.outs in
  let bounds = [| elements1 i64 lo; elements1 i64 hi |] in
  match compute ~shape:[| 1 |] q bounds with
  | Ok r ->
      equal ~msg:"least" (array string) [| want_lo |] r.(0);
      equal ~msg:"greatest" (array string) [| want_hi |] r.(1)
  | Error a -> failf "the ranges refused: %a" Nx_array_support.pp_answer a

let tests =
  [
    group "domains"
      [ test "every kind and dtype as prog.mli states" test_domains ];
    group "bits"
      [
        prop "bits are a one-element array's bytes"
          (Gen.pair (Gen.of_list ~pp:pp_dtype D.all) floats)
          law_bits;
        test "refuses an int outside the dtype's range" test_bits_refuse;
      ];
    group "programs"
      [
        prop "a program holds what v was given" drawn law_program;
        test "v refuses ill-formed programs" test_program_refuses;
        prop "of_node is v over the operands' nodes" one_node law_of_node;
        prop ~examples:singles
          "single is of_node's node where it reads its operands in order"
          one_node law_single_of_node;
        test "single is None but for of_node's programs" test_single;
        prop ~examples:[ identity ]
          "is_identity is one operand, one output, that operand's In" drawn
          law_identity;
        cases "is_identity at named programs"
          ~id:(fun (n, _, _) -> n)
          identities test_identity;
        test "dtype refuses a node past the program" test_dtype_refuses;
      ];
    group "expressions"
      [
        prop "of_exprs is the outputs' distinct nodes, computing as they do"
          (Gen.pair drawn Gen.int) law_of_exprs;
        prop "of_exprs is so for programs of hundreds of nodes"
          (Gen.pair grown Gen.int) law_of_grown_exprs;
        test "of_exprs merges two builds of 15000 nodes"
          test_of_exprs_two_builds;
        test "of_exprs merges 2^17 fresh copies into 17 nodes"
          test_of_exprs_fresh_copies;
        test "each kind function makes its kind's node"
          test_exprs_kind_functions;
        cases "each operator makes the node of the function it names"
          ~id:(fun (n, _, _, _) -> n)
          operators test_operator;
        test "constructors and of_exprs refuse as v does" test_exprs_refuse;
        test "programs take any number of operands and outputs"
          test_many_operands;
      ];
    group "substitution"
      [
        prop "subst keeping every leaf is p lowered as of_exprs lowers it"
          drawn law_subst_keeps;
        prop "subst is so for programs of hundreds of nodes" grown
          law_subst_keeps;
        prop "subst binding an operand is p over its value" binding law_bind;
        test "subst binds an operand to an operand, a coordinate or a constant"
          test_subst_leaves;
        prop "subst offsetting coordinates is p at the offset index" shifting
          law_subst_coords;
        prop "subst with outs is the outputs of p it selects" selecting
          law_subst_outs;
        test "subst refuses what does not fit" test_subst_refuses;
      ];
    group "ranges"
      [
        prop ~count:400 "every node lies within its range over the box"
          ranged_cases law_ranges;
        cases "a cast into bit has bit's range" ~id:(fun (n, _, _) -> n)
          bit_casts test_bit_cast;
      ];
  ]

let () = exit (run "nx_kernel.prog" tests)
