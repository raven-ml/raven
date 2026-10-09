(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Values' facts *)

let placement : type v s d. (v, s, d) t -> d Devices.placement = function
  | Array { at; _ } | Shards { at; _ } -> at
  | Deferred { form; _ } -> form.placement

let dtype : type v s d. (v, s, d) t -> (v, s) dtype = function
  | Array { a; _ } -> Nx_array.dtype a
  | Shards { arrays; _ } -> Nx_array.dtype arrays.(0)
  | Deferred { form; _ } -> form.dtype

(* The layout of [x]'s array, of its first shard, or of the constant. *)
let own_layout : type v s d. (v, s, d) t -> L.t = function
  | Array { a; _ } -> Nx_array.layout a
  | Shards { arrays; _ } -> Nx_array.layout arrays.(0)
  | Deferred { form; _ } -> form.layout

let rank x = L.rank (own_layout x)

(* The tiles along [axis] of a value at [p]: 1 where it is not cut. *)
let tiles p axis =
  Array.fold_left
    (fun n (a, t) -> if a = axis then t else n)
    1
    (Grid.cuts (Devices.grid p))

let dim (type v s d) (x : (v, s, d) t) i =
  let l = own_layout x in
  if i < 0 || i >= L.rank l then
    invalid_arg
      (Printf.sprintf "Prim.dim: axis %d of a value of rank %d" i (L.rank l));
  match x with
  | Array _ | Deferred _ -> L.dim l i
  | Shards { at; _ } -> L.dim l i * tiles at i

let shape x = Array.init (rank x) (dim x)

let form (type v s d) (x : (v, s, d) t) : (v, s, d) form =
  match x with
  | Array { at; a } ->
      { dtype = Nx_array.dtype a; layout = Nx_array.layout a; placement = at }
  | Shards { at; _ } ->
      { dtype = dtype x; layout = L.contiguous (shape x); placement = at }
  | Deferred { form; _ } -> form

let is_constant : type v s d. (v, s, d) t -> bool = function
  | Deferred _ -> true
  | Array _ | Shards _ -> false

let expect (type v s) (dt : (v, s) dtype) (Any x : 'd any) : (v, s, 'd) t =
  match D.equal_witness dt (dtype x) with
  | Some Type.Equal -> x
  | None ->
      invalid_argf "Prim.expect: a %s value where %s was expected"
        (D.name (dtype x))
        (D.name dt)

(* Operations *)

type operands = Operands : 'd any list -> operands
type mapper = { map : 'v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t }
type maker = { make : 'v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t }

let name : type r. r prim -> string = function
  | Map _ -> "Map"
  | Copy _ -> "Copy"
  | Move _ -> "Move"
  | Bitcast _ -> "Bitcast"
  | Place _ -> "Place"
  | Check _ -> "Check"

let load_any (type d) (Plain x : d load) : d any = Any x

let operands : type r. r prim -> operands = function
  | Map { loads; _ } -> Operands (Array.to_list (Array.map load_any loads))
  | Copy x -> Operands [ Any x ]
  | Move (_, x) -> Operands [ Any x ]
  | Bitcast (_, x) -> Operands [ Any x ]
  | Place (_, x) -> Operands [ Any x ]
  | Check { ok; data; _ } -> Operands (Any ok :: data)

let map_load (type d) m (Plain x : d load) : d load = Plain (m.map x)

let map : type r. mapper -> r prim -> r prim =
 fun m op ->
  match op with
  | Map p -> Map { p with loads = Array.map (map_load m) p.loads }
  | Copy x -> Copy (m.map x)
  | Move (mv, x) -> Move (mv, m.map x)
  | Bitcast (dt, x) -> Bitcast (dt, m.map x)
  | Place (p, x) -> Place (p, m.map x)
  | Check c ->
      Check
        {
          c with
          ok = m.map c.ok;
          data = List.map (fun (Any x) -> Any (m.map x)) c.data;
        }

(* Printing *)

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

let any_name (D.Any dt) = D.name dt

let kind : P.node -> string = function
  | In _ -> "In"
  | Coord _ -> "Iota"
  | Const _ -> "Fill"
  | Op1 (Copy, _, _) -> "Copy"
  | Op1 (Unary k, _, _) -> String.capitalize_ascii (unary_name k)
  | Op1 (Cast, _, _) -> "Cast"
  | Op1 (Bitcast, _, _) -> "Bitcast"
  | Op2 (Binary k, _, _) -> String.capitalize_ascii (binary_name k)
  | Op2 (Compare k, _, _) -> String.capitalize_ascii (compare_name k)
  | Op3 (Where, _, _, _) -> "Where"
  | Op3 (Fma, _, _, _) -> "Fma"

let hex bits =
  String.concat ""
    (List.map
       (fun c -> Printf.sprintf "%02x" (Char.code c))
       (List.of_seq (String.to_seq bits)))

(* Node [i] of [p] as an expression of the operands [x0], [x1], …. *)
let rec pp_node p ppf i =
  match P.node p i with
  | In k -> Format.fprintf ppf "x%d" k
  | Coord a -> Format.fprintf ppf "coord%d" a
  | Const (dt, bits) -> Format.fprintf ppf "%s:0x%s" (any_name dt) (hex bits)
  | Op1 (Copy, _, j) -> Format.fprintf ppf "copy(%a)" (pp_node p) j
  | Op1 (Unary k, _, j) ->
      Format.fprintf ppf "%s(%a)" (unary_name k) (pp_node p) j
  | Op1 (Cast, dt, j) ->
      Format.fprintf ppf "cast_%s(%a)" (any_name dt) (pp_node p) j
  | Op1 (Bitcast, dt, j) ->
      Format.fprintf ppf "bitcast_%s(%a)" (any_name dt) (pp_node p) j
  | Op2 (Binary k, j, l) ->
      Format.fprintf ppf "%s(%a, %a)" (binary_name k) (pp_node p) j (pp_node p)
        l
  | Op2 (Compare k, j, l) ->
      Format.fprintf ppf "%s(%a, %a)" (compare_name k) (pp_node p) j (pp_node p)
        l
  | Op3 (Where, j, l, n) ->
      Format.fprintf ppf "where(%a, %a, %a)" (pp_node p) j (pp_node p) l
        (pp_node p) n
  | Op3 (Fma, j, l, n) ->
      Format.fprintf ppf "fma(%a, %a, %a)" (pp_node p) j (pp_node p) l
        (pp_node p) n

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let pp_operand ppf (Any x) =
  Format.fprintf ppf "%s %a at %a"
    (D.name (dtype x))
    pp_shape (shape x) Devices.pp_placement (placement x)

let pp ppf op =
  let (Operands xs) = operands op in
  Format.fprintf ppf "%s" (name op);
  (match op with
  | Map { prog; _ } ->
      Format.fprintf ppf " [%a]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           (pp_node prog))
        (Array.to_list (P.outs prog))
  | _ -> ());
  List.iteri (fun i x -> Format.fprintf ppf " (x%d: %a)" i pp_operand x) xs

(* Rules *)

let same_dtype a b = D.code a = D.code b
let load_shape (type d) (Plain x : d load) = shape x

let load_placement (type d) (Plain x : d load) : d Devices.placement =
  placement x

let load_dtype (type d) (Plain x : d load) : D.any = D.Any (dtype x)

let check_map (type d r) ~by shape prog (outs : (d, r) outs)
    (loads : d load array) =
  let ins = P.ins prog in
  if Array.length loads <> Array.length ins then
    invalid_argf "%s: a program of %d operands over %d loads" by
      (Array.length ins) (Array.length loads);
  Array.iteri
    (fun i l ->
      let (D.Any want) = ins.(i) in
      let (D.Any have) = load_dtype l in
      if not (same_dtype want have) then
        invalid_argf "%s: load %d is %s where the program reads %s" by i
          (D.name have) (D.name want))
    loads;
  let produced = P.outs prog in
  let rec check : type q. int -> (d, q) outs -> unit =
   fun k -> function
     | [] ->
         if k <> Array.length produced then
           invalid_argf "%s: %d results for a program of %d outputs" by k
             (Array.length produced)
     | dt :: rest ->
         if k >= Array.length produced then
           invalid_argf "%s: more results than the program's %d outputs" by
             (Array.length produced);
         let (D.Any want) = P.dtype prog produced.(k) in
         if not (same_dtype want dt) then
           invalid_argf "%s: result %d is %s where the program gives %s" by k
             (D.name dt) (D.name want);
         check (k + 1) rest
  in
  check 0 outs;
  (match L.contiguous shape with
  | _ -> ()
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e);
  Array.iteri
    (fun i l ->
      if load_shape l <> shape then
        invalid_argf "%s: load %d has shape %a, the map %a" by i pp_shape
          (load_shape l) pp_shape shape)
    loads

(* Each operation's route: where it reads its operands, in [operands]'s order,
   and where its results lie. [results] and [prepare] take placements from it
   alone. *)

let map_route (type d) ~by (loads : d load array) : d Route.t =
  if Array.length loads = 0 then { operands = [||]; result = Devices.anywhere }
  else
    Route.route ~by Elementwise
      (Array.map load_placement loads)
      (Array.map load_shape loads)

let rec make_outs : type d r.
    maker -> int -> L.t -> d Devices.placement -> (d, r) outs -> r =
 fun m k layout placement -> function
  | [] -> ()
  | dtype :: rest ->
      let v = m.make k { dtype; layout; placement } in
      (v, make_outs m (k + 1) layout placement rest)

let one_route ~by rule x = Route.route ~by rule [| placement x |] [| shape x |]

let bitcast_rule dt x =
  if D.bits dt > D.bits (dtype x) then Route.Reduce [| rank x - 1 |]
  else Route.Elementwise

(* A bitcast's layout and shape: one width keeps them; a narrower dtype appends
   an axis of the ratio, a wider one removes it where strides allow, and a
   copy's contiguous layout otherwise. *)
let bitcast_layout ~by l (from : D.any) (into : D.any) =
  let (D.Any f) = from in
  let (D.Any t) = into in
  let bf = D.bits f and bt = D.bits t in
  let s = L.shape l and st = L.strides l and o = L.offset l in
  let r = Array.length s in
  if bf = bt then l
  else if bf > bt then begin
    let k = bf / bt in
    if r >= L.max_rank then
      invalid_argf "%s: a narrowing bitcast of rank %d" by r;
    L.v ~offset:(o * k)
      ~strides:(Array.append (Array.map (fun x -> x * k) st) [| 1 |])
      (Array.append s [| k |])
  end
  else begin
    let k = bt / bf in
    if r = 0 || s.(r - 1) <> k then
      invalid_argf "%s: a widening bitcast needs a trailing axis of %d" by k;
    let s' = Array.sub s 0 (r - 1) in
    let viewable =
      L.numel l = 0
      || st.(r - 1) = 1
         && o mod k = 0
         && Array.for_all (fun x -> x mod k = 0) (Array.sub st 0 (r - 1))
    in
    if viewable && L.numel l > 0 then
      L.v ~offset:(o / k)
        ~strides:(Array.map (fun x -> x / k) (Array.sub st 0 (r - 1)))
        s'
    else L.contiguous s'
  end

let results : type r. by:string -> maker -> r prim -> r =
 fun ~by m op ->
  match op with
  | Map { shape; prog; outs; loads } ->
      check_map ~by shape prog outs loads;
      let placement = (map_route ~by loads).result in
      make_outs m 0 (L.contiguous shape) placement outs
  | Copy x ->
      let placement = (one_route ~by Elementwise x).result in
      m.make 0 { dtype = dtype x; layout = L.contiguous (shape x); placement }
  | Move (mv, x) ->
      let s' =
        match Nx_array.Move.shape mv (shape x) with
        | s' -> s'
        | exception Invalid_argument e -> invalid_argf "%s: %s" by e
      in
      let placement = (one_route ~by (Move mv) x).result in
      let layout =
        match L.move mv (form x).layout with
        | Some l -> l
        | None -> L.contiguous s'
      in
      m.make 0 { dtype = dtype x; layout; placement }
  | Bitcast (dt, x) ->
      let layout =
        bitcast_layout ~by (form x).layout (D.Any (dtype x)) (D.Any dt)
      in
      let placement = (one_route ~by (bitcast_rule dt x) x).result in
      m.make 0 { dtype = dt; layout; placement }
  | Place (p, x) ->
      let s = shape x in
      ignore (Devices.window ~by p s 0);
      let layout =
        match (x, Devices.device p) with
        | Array { a; _ }, Some _ -> Nx_array.layout a
        | _ -> L.contiguous s
      in
      m.make 0 { dtype = dtype x; layout; placement = p }
  | Check { ok; data; _ } ->
      let s = shape ok in
      List.iteri
        (fun i (Any x) ->
          if shape x <> s then
            invalid_argf "%s: datum %d has shape %a, the flag %a" by i pp_shape
              (shape x) pp_shape s)
        data

(* One-node programs and maps *)

(* Each domain's one-node programs, by node and operand dtypes: plain data,
   compared structurally. A program built once is not built again, which saves
   a slow-path operation the program's construction (7% of
   dispatch/zeros_like-1). *)
let programs = Domain.DLS.new_key (fun () -> Hashtbl.create 64)

let program node ins =
  let table = Domain.DLS.get programs in
  let key = (node, ins) in
  match Hashtbl.find_opt table key with
  | Some p -> p
  | None ->
      let p = P.of_node ~ins node in
      Hashtbl.add table key p;
      p

let one_node node dt ins loads =
  let shape = match loads.(0) with Plain x -> shape x in
  Map { shape; prog = program node ins; outs = Value.[ dt ]; loads }

let op1 k dt x =
  one_node (P.Op1 (k, D.Any dt, 0)) dt [| D.Any (dtype x) |] [| Plain x |]

let op2 k dt x y =
  let i = D.Any (dtype x) in
  one_node (P.Op2 (k, 0, 1)) dt [| i; i |] [| Plain x; Plain y |]

let op3 k c x y =
  let i = D.Any (dtype x) in
  one_node
    (P.Op3 (k, 0, 1, 2))
    (dtype x)
    [| D.Any (dtype c); i; i |]
    [| Plain c; Plain x; Plain y |]

type placer = {
  place : 'v 's 'd. 'd Devices.placement -> ('v, 's, 'd) t -> ('v, 's, 'd) t;
}

let prepare : type r. by:string -> placer -> r prim -> r prim =
 fun ~by pl op ->
  let one rule x = pl.place (one_route ~by rule x).operands.(0) x in
  match op with
  | Map p ->
      let r = map_route ~by p.loads in
      Map
        {
          p with
          loads =
            Array.mapi
              (fun i (Plain x) -> Plain (pl.place r.operands.(i) x))
              p.loads;
        }
  | Copy x -> Copy (one Elementwise x)
  | Move (mv, x) -> Move (mv, one (Move mv) x)
  | Bitcast (dt, x) -> Bitcast (dt, one (bitcast_rule dt x) x)
  | Place _ | Check _ -> op

let arrays_of : type v s d. (v, s, d) t -> Nx_array.any array = function
  | Array { a; _ } -> [| Nx_array.Any a |]
  | Shards { arrays; _ } -> Array.map (fun a -> Nx_array.Any a) arrays
  | Deferred _ -> invalid_arg "Prim.arrays: a constant has no arrays"

let rec arrays_outs : type d q. (d, q) outs -> q -> Nx_array.any array list =
 fun outs r ->
  match (outs, r) with
  | [], () -> []
  | _ :: rest, (v, r) -> arrays_of v :: arrays_outs rest r

let arrays : type r. r prim -> r -> Nx_array.any array array =
 fun op r ->
  match op with
  | Map { outs; _ } -> Array.of_list (arrays_outs outs r)
  | Copy _ -> [| arrays_of r |]
  | Move _ -> [| arrays_of r |]
  | Bitcast _ -> [| arrays_of r |]
  | Place _ -> [| arrays_of r |]
  | Check _ -> [||]
