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
   node read twice is one value. *)
let exprs d =
  let e = Array.make (Array.length d.nodes) (Ex (P.E.coord 0)) in
  Array.iteri
    (fun i nd ->
      e.(i) <-
        (match nd with
        | P.In k ->
            let (D.Any dt) = d.ins.(k) in
            Ex (P.E.input k dt)
        | Coord a -> Ex (P.E.coord a)
        | Const (D.Any dt, b) -> Ex (P.E.const dt (element dt b))
        | Op1 (k, D.Any dt, a) ->
            let (Ex x) = e.(a) in
            Ex (P.E.op1 k dt x)
        | Op2 (k, a, b) -> (
            let (Ex x) = e.(a) in
            let (Ex y) = e.(b) in
            match k with
            | Binary k -> Ex (P.E.binary k x (like x y))
            | Compare k -> Ex (P.E.compare k x (like x y)))
        | Op3 (k, a, b, c) ->
            let (Ex c') = e.(a) in
            let (Ex x) = e.(b) in
            let (Ex y) = e.(c) in
            Ex (P.E.op3 k c' x (like x y))))
    d.nodes;
  Array.map (fun (Ex x) -> P.E.any x) e

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

(* [d] without the nodes its outputs do not reach, whose kinds a library
   may decline. *)
let pruned d =
  let keep = Array.make (Array.length d.nodes) false in
  let rec visit i =
    if not keep.(i) then begin
      keep.(i) <- true;
      match d.nodes.(i) with
      | P.In _ | Coord _ | Const _ -> ()
      | Op1 (_, _, a) -> visit a
      | Op2 (_, a, b) -> visit a; visit b
      | Op3 (_, a, b, c) -> visit a; visit b; visit c
    end
  in
  Array.iter visit d.outs;
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
let random_elements rand (D.Any dt as d) n =
  let byte () = Char.chr (Random.State.int rand 256) in
  let one () =
    if D.equal dt D.Bool then String.make 1 (Char.chr (Random.State.int rand 2))
    else String.init (max 1 (D.bits dt / 8)) (fun _ -> byte ())
  in
  Elements.of_elements d [| 2; 3; 2 |] (Array.init n (fun _ -> one ()))

(* [p]'s outputs over [loads] at the shape [2; 3; 2], computed by nx.cpu, as
   their elements' bits. *)
let compute (p : P.t) loads =
  let dst o =
    let (D.Any dt) = P.dtype p o in
    A.Any (A.create Rig.host dt [| 2; 3; 2 |])
  in
  let dsts = Array.map dst (Iarray.to_array p.outs) in
  let loads = Array.map (fun (A.Any x) -> Nx_kernel.Plain x) loads in
  match Nx_cpu.map p ~dsts loads with
  | Done -> Ok (Array.map Elements.elements dsts)
  | answer -> Error answer

let outputs = Testable.(result (array (array string)) Elements.answer)

(* Binding *)

(* An expression drawn over [ins], its last node cast to [dt]. *)
let ending ins choices dt =
  let d = build ins choices 1 in
  let last = Array.length d.nodes - 1 in
  let d = { d with nodes = Array.append d.nodes [| P.Op1 (Cast, dt, last) |] } in
  (exprs d).(last + 1)

let binding =
  let open Gen in
  with_pp
    (fun ppf (d, k, _, _) -> Format.fprintf ppf "%a, operand %d" pp_drawn d k)
    (let+ d = drawn
     and+ k = int_range 0 2
     and+ choices = array ~size:(int_range 1 6) (int_range 0 1_000_000)
     and+ seed = int in
     (d, k, choices, seed))

(* [bind p k x]'s nodes as prog.mli states them: [q]'s, [x] lowered over the
   result's operands, then [p]'s but [In k], renumbered. *)
let bound_nodes (p : P.t) k (q : P.t) =
  let out = Iarray.get q.outs 0 in
  let index = Array.make (Iarray.length p.nodes) 0 in
  let nodes = ref (List.rev (Iarray.to_list q.nodes)) in
  let count = ref (Iarray.length q.nodes) in
  Iarray.iteri
    (fun i nd ->
      let at j = index.(j) in
      match (nd : P.node) with
      | In j when j = k -> index.(i) <- out
      | _ ->
          let nd : P.node =
            match nd with
            | In j when j > k -> In (j - 1)
            | Op1 (o, dt, x) -> Op1 (o, dt, at x)
            | Op2 (o, x, y) -> Op2 (o, at x, at y)
            | Op3 (o, x, y, z) -> Op3 (o, at x, at y, at z)
            | nd -> nd
          in
          nodes := nd :: !nodes;
          index.(i) <- !count;
          incr count)
    p.nodes;
  (Array.of_list (List.rev !nodes), Array.map (fun o -> index.(o)) (Iarray.to_array p.outs))

(* Checks [bind p k x] against prog.mli: its operands, its nodes and outputs,
   v's program of them, and its outputs' bits on nx.cpu against [p]'s over
   [x]'s value, over operands drawn from [seed]. *)
let check_bind (p : P.t) k x seed =
  let ins = Iarray.to_array p.ins in
  let others = Array.of_list (List.filteri (fun i _ -> i <> k) (Array.to_list ins)) in
  let b = P.bind p k x in
  let q = P.of_exprs ~ins:(Iarray.of_array others) [| x |] in
  let nodes, outs = bound_nodes p k q in
  equal ~msg:"nodes" (array node) nodes (Iarray.to_array b.nodes);
  equal ~msg:"outs" (array int) outs (Iarray.to_array b.outs);
  equal ~msg:"ins" (list string) (Array.to_list (Array.map name others))
    (List.map name (Iarray.to_list b.ins));
  equal ~msg:"as v builds it" bool true
    (b = P.v ~ins:b.ins b.nodes ~outs:b.outs);
  let rand = Random.State.make [| seed |] in
  let loads = Array.map (fun t -> random_elements rand t 12) others in
  let computed = compute q loads in
  match computed with
  | Error _ -> false
  | Ok out ->
      let given = Elements.of_elements ins.(k) [| 2; 3; 2 |] out.(0) in
      let all =
        Array.init (Array.length ins) (fun i ->
            if i = k then given else loads.(if i < k then i else i - 1))
      in
      equal ~msg:"outputs' bits" outputs (compute p all) (compute b loads);
      true

let law_bind (d, k, choices, seed) =
  assume (k < Array.length d.ins);
  let others = Array.of_list (List.filteri (fun i _ -> i <> k) (Array.to_list d.ins)) in
  cover "p reads operand k" (Array.mem (P.In k) d.nodes);
  cover "an operand after k" (k < Array.length d.ins - 1);
  cover "k is 0" (k = 0);
  cover "x computed" (check_bind (program d) k (ending others choices d.ins.(k)) seed)

(* Expressions whose value is an operand, a coordinate or a constant, at k =
   0, with p reading or outputting In k. *)
let test_bind_leaves () =
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 and i64 = D.Any D.Int64 in
  let p ins nodes outs =
    P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes) ~outs:(Iarray.of_array outs)
  in
  let sum = p [| f32; f32 |] [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |] [| 2 |] in
  equal ~msg:"an operand" bool true
    (check_bind sum 0 P.E.(any (input 0 D.Float32)) 1);
  let scaled =
    p [| i64; f32 |] [| P.In 0; In 1; Op1 (Cast, f32, 0); Op2 (Binary Mul, 2, 1) |] [| 3 |]
  in
  equal ~msg:"a coordinate" bool true (check_bind scaled 0 P.E.(any (coord 1)) 2);
  let passed = p [| f32; f64 |] [| P.In 0; In 1 |] [| 0; 1 |] in
  equal ~msg:"a constant, output as p's In k" bool true
    (check_bind passed 0 P.E.(any (const D.Float32 2.5)) 3)

let test_bind_refuses () =
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 in
  let p = P.of_exprs ~ins:[| f32; f64 |] [| P.E.(any (input 0 D.Float32)) |] in
  let refuses ~msg k x =
    raises_match ~msg Exn.invalid_arg (fun () -> P.bind p k x)
  in
  refuses ~msg:"an operand past ins" 2 P.E.(any (const D.Float32 1.));
  refuses ~msg:"a negative operand" (-1) P.E.(any (const D.Float32 1.));
  refuses ~msg:"an expression of another dtype" 1 P.E.(any (const D.Float32 1.));
  refuses ~msg:"a read of an operand the result lacks" 1
    P.E.(any (cast D.Float64 (input 1 D.Float32)));
  refuses ~msg:"a read of an operand at another dtype" 1
    P.E.(any (input 0 D.Float64));
  ignore (P.bind p 1 P.E.(any (const D.Float64 1.)))

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
  let want = compute (program (pruned d)) loads in
  equal ~msg:"outputs' bits" outputs want (compute q loads);
  q

let law_of_exprs (d, seed) =
  let q = check_exprs d seed in
  cover "outputs of two builds" (Array.length d.outs >= 2);
  cover "fewer nodes than drawn" (Iarray.length q.nodes < Array.length d.nodes)

(* Programs grown node by node, most of them past the walk's first 128 nodes:
   each node is an operand, an axis, a constant, a copy of an earlier node's
   structure, or a kind over earlier nodes of the dtypes it takes. Constants
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
  let nouts = min nouts (P.max_operands - Array.length ins) in
  let outs = Array.init nouts (fun _ -> if int 2 = 0 then n - 1 else int n) in
  { ins; nodes; types; outs }

let grown =
  let open Gen in
  with_pp pp_drawn
    (let+ ins = array ~size:(int_range 0 3) (of_list ~pp:pp_dtype base_dtypes)
     and+ n = int_range 1 400
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

let test_of_exprs_deep () =
  (* [y + y - y] from [y], 5000 times: a tree of 3^5000 leaves, a DAG of
     10001 nodes, and the operand back. *)
  let y = ref (P.E.input 0 D.Uint64) in
  for _ = 1 to 5000 do
    y := P.E.(!y + !y - !y)
  done;
  let q = P.of_exprs ~ins:[| D.Any D.Uint64 |] [| P.E.any !y |] in
  equal ~msg:"nodes" int 10001 (Iarray.length q.nodes);
  let input = A.of_array D.Uint64 [| 2; 3; 2 |] (Array.init 12 Int64.of_int) in
  equal ~msg:"the operand" outputs
    (Ok [| Elements.elements (A.Any input) |])
    (compute q [| A.Any input |])

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

(* Each kind and dtype comes through a lowering as the node it was. *)
let test_of_exprs_kinds () =
  let f32 = D.Any D.Float32 and u64 = D.Any D.Uint64 and b = D.Any D.Bool in
  let ins = [| f32; f32; b; u64 |] in
  let one out nd =
    let q = P.of_exprs ~ins:(Iarray.of_array ins) [| P.E.any out |] in
    let last = Iarray.length q.nodes - 1 in
    equal ~msg:"output" (array int) [| last |] (Iarray.to_array q.outs);
    equal ~msg:"node" node nd (Iarray.get q.nodes last)
  in
  let x = P.E.input 0 D.Float32 and y = P.E.input 1 D.Float32 in
  let c = P.E.input 2 D.Bool and w = P.E.input 3 D.Uint64 in
  List.iter
    (fun k ->
      let (D.Any dt as d) = match k with P.Bitcast -> D.Any D.Int32 | _ -> f32 in
      match P.E.op1 k dt x with
      | e -> one e (P.Op1 (k, d, 0))
      | exception Invalid_argument _ -> ())
    op1s;
  List.iter
    (fun k ->
      match k with
      | P.Binary (Idiv | And | Or | Xor | Threefry as k) ->
          one (P.E.binary k w w) (P.Op2 (Binary k, 0, 0))
      | Binary k -> one (P.E.binary k x y) (P.Op2 (Binary k, 0, 1))
      | Compare k -> one (P.E.compare k x y) (P.Op2 (Compare k, 0, 1)))
    op2s;
  one (P.E.op3 Where c x y) (P.Op3 (Where, 0, 1, 2));
  one (P.E.op3 Fma x y x) (P.Op3 (Fma, 0, 1, 0));
  one (P.E.const D.Float64 (-0.)) (P.Const (D.Any D.Float64, P.bits D.Float64 (-0.)));
  one (P.E.coord 3) (P.Coord 3)

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
  List.iter2 u unaries
    P.E.
      [
        neg; recip; abs; sign; sqrt; exp; exp2; log; log2; log1p; expm1; sin; cos;
        tan; asin; acos; atan; sinh; cosh; tanh; erf; floor; ceil; round; trunc;
      ];
  List.iter2 b
    P.[ Add; Sub; Mul; Fdiv; Mod; Pow; Atan2; Maximum; Minimum ]
    P.E.[ add; sub; mul; fdiv; rem; pow; atan2; maximum; minimum ];
  List.iter2 cmp compares P.E.[ equal; not_equal; less; less_equal ];
  same "threefry" (P.E.any (P.E.threefry w w)) (P.E.any (P.E.binary Threefry w w));
  same "logxor" (P.E.any (P.E.logxor w w)) (P.E.any (P.E.binary Xor w w));
  same "&&" (P.E.any P.E.(c && c)) (P.E.any (P.E.binary And c c));
  same "||" (P.E.any P.E.(c || c)) (P.E.any (P.E.binary Or c c));
  same "/ on floats" (P.E.any P.E.(x / y)) (P.E.any (P.E.binary Fdiv x y));
  same "/ on integers" (P.E.any P.E.(w / w)) (P.E.any (P.E.binary Idiv w w));
  same "copy" (P.E.any (P.E.copy x)) (P.E.any (P.E.op1 Copy D.Float32 x));
  same "cast" (P.E.any (P.E.cast D.Int8 x)) (P.E.any (P.E.op1 Cast D.Int8 x));
  same "bitcast" (P.E.any (P.E.bitcast D.Int32 x)) (P.E.any (P.E.op1 Bitcast D.Int32 x));
  same "where" (P.E.any (P.E.where c x y)) (P.E.any (P.E.op3 Where c x y));
  same "fma" (P.E.any (P.E.fma x y x)) (P.E.any (P.E.op3 Fma x y x))

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
  raises_match ~msg:"an axis past max_rank" Exn.invalid_arg (fun () ->
      P.E.coord Nx_array.Layout.max_rank);
  raises_match ~msg:"a negative axis" Exn.invalid_arg (fun () -> P.E.coord (-1));
  raises_match ~msg:"an operand past max_operands" Exn.invalid_arg (fun () ->
      P.E.input P.max_operands D.Float32);
  raises_match ~msg:"an operand past int32" Exn.invalid_arg (fun () ->
      P.E.input (1 lsl 32) D.Float32);
  raises_match ~msg:"a negative operand" Exn.invalid_arg (fun () ->
      P.E.input (-1) D.Float32);
  let x = P.E.any (P.E.input 0 D.Float32) in
  raises_match ~msg:"no output" Exn.invalid_arg (fun () ->
      P.of_exprs ~ins:[| f32 |] [||]);
  raises_match ~msg:"more operands and outputs than max_operands"
    Exn.invalid_arg (fun () ->
      P.of_exprs ~ins:(Iarray.init P.max_operands (fun _ -> f32)) [| x |]);
  raises_match ~msg:"an operand past ins" Exn.invalid_arg (fun () ->
      P.of_exprs ~ins:[||] [| x |]);
  raises_match ~msg:"an operand of another dtype" Exn.invalid_arg (fun () ->
      P.of_exprs ~ins:[| i32 |] [| x |])

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
  (* No code past the kinds named above. *)
  equal ~msg:"NX_OP1_COUNT" int (code "NX_OP1_COUNT") (List.length op1s);
  equal ~msg:"NX_OP2_COUNT" int (code "NX_OP2_COUNT") (List.length op2s);
  equal ~msg:"NX_OP3_COUNT" int (code "NX_OP3_COUNT") (List.length P.[ Where; Fma ]);
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
        prop "bind is p over x's value, its nodes x's then p's" binding
          law_bind;
        test "bind takes an operand, a coordinate or a constant"
          test_bind_leaves;
        test "bind refuses what is no binding" test_bind_refuses;
      ];
    group "expressions"
      [
        prop "of_exprs is the outputs' distinct nodes, computing as they do"
          (Gen.pair drawn Gen.int) law_of_exprs;
        prop "of_exprs is so for programs of hundreds of nodes"
          (Gen.pair grown Gen.int) law_of_grown_exprs;
        test "of_exprs lowers a DAG of exponential tree size"
          test_of_exprs_deep;
        test "of_exprs merges two builds of 15000 nodes"
          test_of_exprs_two_builds;
        test "of_exprs keeps each kind" test_of_exprs_kinds;
        test "each kind function makes its kind's node"
          test_exprs_kind_functions;
        test "constructors and of_exprs refuse as v does" test_exprs_refuse;
      ];
  ]

let () = exit (run "nx_kernel.prog" tests)
