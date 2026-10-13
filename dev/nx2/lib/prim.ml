(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module Shape = Nx_kernel.Shape

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Values' facts *)

let at : type v s d. (v, s, d) t -> d Devices.placement option = function
  | Array { at; _ } | Shards { at; _ } | Donated { at; _ } -> Some at
  | Deferred { form; _ } | Traced { form; _ } -> form.placement

let placement : type v s d. (v, s, d) t -> d Devices.placement = function
  | Array { at; _ } | Shards { at; _ } | Donated { at; _ } -> at
  | Traced { form = { placement = Some p; _ }; _ } -> p
  | Deferred _ | Traced _ -> invalid_arg "Prim.placement: a value of every set"

let dtype : type v s d. (v, s, d) t -> (v, s) dtype = function
  | Array { a; _ } -> Nx_array.dtype a
  | Shards { arrays; _ } | Donated { arrays; _ } ->
      Nx_array.dtype (Iarray.get arrays 0)
  | Deferred { form; _ } | Traced { form; _ } -> form.dtype

(* The layout of [x]'s array, of its first shard, or of its form. *)
let own_layout : type v s d. (v, s, d) t -> L.t = function
  | Array { a; _ } -> Nx_array.layout a
  | Shards { arrays; _ } | Donated { arrays; _ } ->
      Nx_array.layout (Iarray.get arrays 0)
  | Deferred { form; _ } | Traced { form; _ } -> form.layout

let rank x = L.rank (own_layout x)

let death : type v s d. (v, s, d) t -> death = function
  | Array r -> r.dead
  | Shards r -> r.dead
  | Donated r -> (
      (* A handle dies with its donor, whichever handle was consumed. *)
      match Atomic.Loc.get r.donor with
      | Died _ as died -> died
      | Live -> r.spent)
  | Deferred _ | Traced _ -> Live

let alive ~by i x =
  match death x with
  | Live -> ()
  | Died why -> invalid_argf "%s: operand %d was donated to %s" by (i + 1) why

let arrays : type v s d. (v, s, d) t -> (v, s) Nx_array.t iarray = function
  | Array { a; _ } -> [| a |]
  | Shards { arrays; _ } | Donated { arrays; _ } -> arrays
  | Deferred _ -> invalid_arg "Prim.arrays: a constant has no arrays"
  | Traced _ -> invalid_arg "Prim.arrays: a traced value has no arrays"

let of_arrays (type v s d) (p : d Devices.placement)
    (arrays : (v, s) Nx_array.t iarray) : (v, s, d) t =
  if Iarray.length arrays = 1 then
    Array { at = p; a = Iarray.get arrays 0; dead = Live }
  else Shards { at = p; arrays; dead = Live }

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
  | Array _ | Deferred _ | Traced _ -> L.dim l i
  | Shards { at; _ } | Donated { at; _ } -> L.dim l i * tiles at i

let shape (type v s d) (x : (v, s, d) t) =
  match x with
  | Array _ | Deferred _ | Traced _ -> L.shape (own_layout x)
  | Shards _ | Donated _ -> Array.init (rank x) (fun i -> dim x i)

let has_shape x s =
  let rec go i = i = Array.length s || (dim x i = s.(i) && go (i + 1)) in
  rank x = Array.length s && go 0

let rec layouts_equal la lb i =
  i = L.rank la || (L.dim la i = L.dim lb i && layouts_equal la lb (i + 1))

let rec dims_equal x y i =
  i = rank x || (dim x i = dim y i && dims_equal x y (i + 1))

let same_shape (type v s w r d) (x : (v, s, d) t) (y : (w, r, d) t) =
  match (x, y) with
  | Array { a; _ }, Array { a = b; _ } ->
      let la = Nx_array.layout a and lb = Nx_array.layout b in
      L.rank la = L.rank lb && layouts_equal la lb 0
  | _ -> rank x = rank y && dims_equal x y 0

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let merge s s' =
  let r = max (Array.length s) (Array.length s') in
  let at s i =
    let k = i - (r - Array.length s) in
    if k < 0 then 1 else s.(k)
  in
  let out = Array.make r 1 in
  let rec go i =
    if i = r then Ok out
    else
      let a = at s i and b = at s' i in
      if a = b || b = 1 then (
        out.(i) <- a;
        go (i + 1))
      else if a = 1 then (
        out.(i) <- b;
        go (i + 1))
      else Error (i, a, b)
  in
  go 0

let broadcast_shape ~by s s' =
  match merge s s' with
  | Ok s'' -> s''
  | Error _ ->
      invalid_argf "%s: shapes %a and %a do not broadcast" by pp_shape s
        pp_shape s'

(* The form of a value of [dtype] laid out as [layout] at [placement]: a value
   on several devices, whose arrays are each its window, is laid out as the
   C-contiguous whole. *)
let of_layout (type v s d) (dtype : (v, s) dtype) layout
    (placement : d Devices.placement option) : (v, s, d) form =
  match placement with
  | Some p
    when Grid.count (Devices.grid p) > 1
         && not (L.is_contiguous layout && L.offset layout = 0) ->
      { dtype; layout = L.contiguous (L.shape layout); placement }
  | Some _ | None -> { dtype; layout; placement }

let form (type v s d) (x : (v, s, d) t) : (v, s, d) form =
  match x with
  | Array { at; a; _ } ->
      {
        dtype = Nx_array.dtype a;
        layout = Nx_array.layout a;
        placement = Some at;
      }
  | Donated { at; arrays; _ } when Iarray.length arrays = 1 ->
      let a = Iarray.get arrays 0 in
      {
        dtype = Nx_array.dtype a;
        layout = Nx_array.layout a;
        placement = Some at;
      }
  | Shards { at; _ } | Donated { at; _ } ->
      of_layout (dtype x) (L.contiguous (shape x)) (Some at)
  | Deferred { form; _ } | Traced { form; _ } -> form

let is_constant : type v s d. (v, s, d) t -> bool = function
  | Deferred _ -> true
  | Array _ | Shards _ | Donated _ | Traced _ -> false

let expect (type v s) (dt : (v, s) dtype) (Any x : 'd any) : (v, s, 'd) t =
  match D.equal_witness dt (dtype x) with
  | Some Type.Equal -> x
  | None ->
      invalid_argf "Prim.expect: a %s value where %s was expected"
        (D.name (dtype x))
        (D.name dt)

(* Operations *)

type operands = Operands : 'd any list -> operands

(* Transforms and factorisations: their routines and operands *)

let routine_name : type d r. (d, r) linalg -> string = function
  | Cholesky _ -> "Cholesky"
  | Lu _ -> "Lu"
  | Qr _ -> "Qr"
  | Svd _ | Svd_values _ -> "Svd"
  | Eigh _ | Eigh_values _ -> "Eigh"
  | Eig _ | Eig_values _ -> "Eig"
  | Solve_triangular _ -> "Solve_triangular"

let linalg_operands : type d r. (d, r) linalg -> d any list = function
  | Cholesky { a; _ } -> [ Any a ]
  | Lu a -> [ Any a ]
  | Qr { a; _ } -> [ Any a ]
  | Svd { a; _ } -> [ Any a ]
  | Svd_values a -> [ Any a ]
  | Eigh a -> [ Any a ]
  | Eigh_values a -> [ Any a ]
  | Eig a -> [ Any a ]
  | Eig_values a -> [ Any a ]
  | Solve_triangular { a; b; _ } -> [ Any a; Any b ]

let fft_axes : type d r. (d, r) fft -> int array = function
  | C2c { axes; _ } | R2c { axes; _ } | C2r { axes; _ } -> axes

let fft_operand : type d r. (d, r) fft -> d any = function
  | C2c { x; _ } -> Any x
  | R2c { x; _ } -> Any x
  | C2r { x; _ } -> Any x

let map_fft : type d r.
    ('v 's. ('v, 's, d) t -> ('v, 's, d) t) -> (d, r) fft -> (d, r) fft =
 fun m -> function
  | C2c f -> C2c { f with x = m f.x }
  | R2c f -> R2c { f with x = m f.x }
  | C2r f -> C2r { f with x = m f.x }

let map_linalg : type d r.
    ('v 's. ('v, 's, d) t -> ('v, 's, d) t) -> (d, r) linalg -> (d, r) linalg
    =
 fun m -> function
  | Cholesky l -> Cholesky { l with a = m l.a }
  | Lu a -> Lu (m a)
  | Qr l -> Qr { l with a = m l.a }
  | Svd l -> Svd { l with a = m l.a }
  | Svd_values a -> Svd_values (m a)
  | Eigh a -> Eigh (m a)
  | Eigh_values a -> Eigh_values (m a)
  | Eig a -> Eig (m a)
  | Eig_values a -> Eig_values (m a)
  | Solve_triangular l -> Solve_triangular { l with a = m l.a; b = m l.b }

let name : type r. r prim -> string = function
  | Map _ -> "Map"
  | Reduce _ -> "Reduce"
  | Scan _ -> "Scan"
  | Gather _ -> "Gather"
  | Scatter _ -> "Scatter"
  | Sort _ -> "Sort"
  | Fold _ -> "Fold"
  | Fft _ -> "Fft"
  | Linalg _ -> "Linalg"
  | Assemble _ -> "Assemble"
  | Contract _ -> "Contract"
  | Copy _ -> "Copy"
  | Move _ -> "Move"
  | Bitcast _ -> "Bitcast"
  | Place _ -> "Place"
  | Check _ -> "Check"

let load_any (type d) (l : d load) : d any =
  match l with Plain x -> Any x | Padded { x; _ } -> Any x

let load_dtype (type d) (l : d load) =
  let (Any x) = load_any l in
  D.Any (dtype x)

let is_padded (type d) (l : d load) =
  match l with Plain _ -> false | Padded _ -> true

let operands : type r. r prim -> operands = function
  | Map { loads; _ } -> Operands (Array.to_list (Array.map load_any loads))
  | Reduce { loads; _ } -> Operands (Array.to_list (Array.map load_any loads))
  | Scan { loads; _ } -> Operands (Array.to_list (Array.map load_any loads))
  | Gather { idx; x; _ } -> Operands [ Any idx; Any x ]
  | Scatter { idx; updates; into; _ } ->
      Operands [ Any idx; Any updates; Any into ]
  | Sort { x; _ } -> Operands [ Any x ]
  | Fold { x; _ } -> Operands [ Any x ]
  | Fft f -> Operands [ fft_operand f ]
  | Linalg l -> Operands (linalg_operands l)
  | Assemble { pieces; _ } -> Operands (List.map (fun (_, x) -> Any x) pieces)
  | Contract { a; b; init; _ } ->
      Operands
        (Any a :: Any b :: Option.to_list (Option.map (fun i -> Any i) init))
  | Copy x -> Operands [ Any x ]
  | Move (_, x) -> Operands [ Any x ]
  | Bitcast (_, x) -> Operands [ Any x ]
  | Place (_, x) -> Operands [ Any x ]
  | Check { ok; data; _ } -> Operands (Any ok :: data)

let iteri_loads : type d.
    ('v 's. int -> ('v, 's, d) t -> unit) -> d load array -> unit =
 fun f loads ->
  for i = 0 to Array.length loads - 1 do
    let (Any x) = load_any loads.(i) in
    f i x
  done

let iteri : type r. ('v 's 'd. int -> ('v, 's, 'd) t -> unit) -> r prim -> unit
    =
 fun f op ->
  match op with
  | Map { loads; _ } -> iteri_loads f loads
  | Reduce { loads; _ } -> iteri_loads f loads
  | Scan { loads; _ } -> iteri_loads f loads
  | Gather { idx; x; _ } ->
      f 0 idx;
      f 1 x
  | Scatter { idx; updates; into; _ } ->
      f 0 idx;
      f 1 updates;
      f 2 into
  | Sort { x; _ } -> f 0 x
  | Fold { x; _ } -> f 0 x
  | Fft fft ->
      let (Any x) = fft_operand fft in
      f 0 x
  | Linalg l -> List.iteri (fun i (Any x) -> f i x) (linalg_operands l)
  | Assemble { pieces; _ } -> List.iteri (fun i (_, x) -> f i x) pieces
  | Contract { a; b; init; _ } ->
      f 0 a;
      f 1 b;
      Option.iter (f 2) init
  | Copy x -> f 0 x
  | Move (_, x) -> f 0 x
  | Bitcast (_, x) -> f 0 x
  | Place (_, x) -> f 0 x
  | Check { ok; data; _ } ->
      f 0 ok;
      List.iteri (fun i (Any x) -> f (i + 1) x) data

let rec loads_exist : type d.
    ('v 's. ('v, 's, d) t -> bool) -> d load array -> int -> bool =
 fun f loads i ->
  i < Array.length loads
  &&
  let (Any x) = load_any loads.(i) in
  f x || loads_exist f loads (i + 1)

let exists : type r. ('v 's 'd. ('v, 's, 'd) t -> bool) -> r prim -> bool =
 fun f op ->
  match op with
  | Map { loads; _ } -> loads_exist f loads 0
  | Reduce { loads; _ } -> loads_exist f loads 0
  | Scan { loads; _ } -> loads_exist f loads 0
  | Gather { idx; x; _ } -> f idx || f x
  | Scatter { idx; updates; into; _ } -> f idx || f updates || f into
  | Sort { x; _ } -> f x
  | Fold { x; _ } -> f x
  | Fft fft ->
      let (Any x) = fft_operand fft in
      f x
  | Linalg l -> List.exists (fun (Any x) -> f x) (linalg_operands l)
  | Assemble { pieces; _ } -> List.exists (fun (_, x) -> f x) pieces
  | Contract { a; b; init; _ } ->
      f a || f b || Option.fold ~none:false ~some:f init
  | Copy x -> f x
  | Move (_, x) -> f x
  | Bitcast (_, x) -> f x
  | Place (_, x) -> f x
  | Check { ok; data; _ } -> f ok || List.exists (fun (Any x) -> f x) data

let map : type r.
    ('v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t) -> r prim -> r prim =
 fun m op ->
  let map_load (type d) (l : d load) : d load =
    match l with
    | Plain x -> Plain (m x)
    | Padded p -> Padded { p with x = m p.x }
  in
  match op with
  | Map p -> Map { p with loads = Array.map map_load p.loads }
  | Reduce p -> Reduce { p with loads = Array.map map_load p.loads }
  | Scan p -> Scan { p with loads = Array.map map_load p.loads }
  | Gather g -> Gather { g with idx = m g.idx; x = m g.x }
  | Scatter s ->
      Scatter { s with idx = m s.idx; updates = m s.updates; into = m s.into }
  | Sort s -> Sort { s with x = m s.x }
  | Fold f -> Fold { f with x = m f.x }
  | Fft f -> Fft (map_fft m f)
  | Linalg l -> Linalg (map_linalg m l)
  | Assemble a ->
      Assemble { a with pieces = List.map (fun (r, x) -> (r, m x)) a.pieces }
  | Contract c ->
      Contract { c with a = m c.a; b = m c.b; init = Option.map m c.init }
  | Copy x -> Copy (m x)
  | Move (mv, x) -> Move (mv, m x)
  | Bitcast (dt, x) -> Bitcast (dt, m x)
  | Place (p, x) -> Place (p, m x)
  | Check c ->
      Check
        {
          c with
          ok = m c.ok;
          data = List.map (fun (Any x) -> Any (m x)) c.data;
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

(* A program's expressions over the operands [x0], [x1], …, of the nodes
   [roots] reads. A node other than a leaf read more than once, by nodes or by
   [roots], is [nK] where read, defined once as [nK = …; ] by [bindings]: the
   text is linear in the program. *)
type printer = {
  bindings : Format.formatter -> unit -> unit;
  expr : Format.formatter -> int -> unit;
}

let printer (p : P.t) roots =
  let reads = Array.make (Iarray.length p.nodes) 0 in
  let read j = reads.(j) <- reads.(j) + 1 in
  for i = 0 to Iarray.length p.nodes - 1 do
    match Iarray.get p.nodes i with
    | In _ | Coord _ | Const _ -> ()
    | Op1 (_, _, j) -> read j
    | Op2 (_, j, l) ->
        read j;
        read l
    | Op3 (_, j, l, n) ->
        read j;
        read l;
        read n
  done;
  List.iter read roots;
  let shared i =
    reads.(i) > 1
    && match Iarray.get p.nodes i with
       | In _ | Coord _ | Const _ -> false
       | _ -> true
  in
  let rec expr ppf i =
    if shared i then Format.fprintf ppf "n%d" i else body ppf i
  and body ppf i =
    match Iarray.get p.nodes i with
    | In k -> Format.fprintf ppf "x%d" k
    | Coord a -> Format.fprintf ppf "coord%d" a
    | Const (dt, bits) -> Format.fprintf ppf "%s:0x%s" (any_name dt) (hex bits)
    | Op1 (Copy, _, j) -> Format.fprintf ppf "copy(%a)" expr j
    | Op1 (Unary k, _, j) -> Format.fprintf ppf "%s(%a)" (unary_name k) expr j
    | Op1 (Cast, dt, j) -> Format.fprintf ppf "cast_%s(%a)" (any_name dt) expr j
    | Op1 (Bitcast, dt, j) ->
        Format.fprintf ppf "bitcast_%s(%a)" (any_name dt) expr j
    | Op2 (Binary k, j, l) ->
        Format.fprintf ppf "%s(%a, %a)" (binary_name k) expr j expr l
    | Op2 (Compare k, j, l) ->
        Format.fprintf ppf "%s(%a, %a)" (compare_name k) expr j expr l
    | Op3 (Where, j, l, n) ->
        Format.fprintf ppf "where(%a, %a, %a)" expr j expr l expr n
    | Op3 (Fma, j, l, n) ->
        Format.fprintf ppf "fma(%a, %a, %a)" expr j expr l expr n
  in
  let bindings ppf () =
    for i = 0 to Iarray.length p.nodes - 1 do
      if shared i then Format.fprintf ppf " n%d = %a;" i body i
    done
  in
  { bindings; expr }

let pp_operand ppf (Any x) =
  let pp_at ppf = function
    | Some p -> Format.fprintf ppf "at %a" Devices.pp_placement p
    | None -> Format.pp_print_string ppf "of every set"
  in
  Format.fprintf ppf "%s %a %a"
    (D.name (dtype x))
    pp_shape (shape x) pp_at (at x)

let reduction : type d a. (d, a) reduction -> Nx_kernel.reduction * int =
  function
  | Monoid (m, k, _) -> (Monoid m, k)
  | Moments (k, _) -> (Moments, k)
  | Arg (e, k, _) -> (Arg e, k)

let reduction_name : Nx_kernel.reduction -> string = function
  | Monoid Sum -> "Sum"
  | Monoid Prod -> "Prod"
  | Monoid Max -> "Max"
  | Monoid Min -> "Min"
  | Monoid Logsumexp -> "Logsumexp"
  | Moments -> "Moments"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"

let rec reductions_list : type d r.
    (d, r) reductions -> (Nx_kernel.reduction * int) list = function
  | [] -> []
  | r :: rest -> reduction r :: reductions_list rest

let reductions rs = Array.of_list (reductions_list rs)

let pp_reduced (prog : P.t) (pr : printer) ppf (r, k) =
  Format.fprintf ppf "%s %a" (reduction_name r) pr.expr
    (Iarray.get prog.outs k)

let reduced_roots (prog : P.t) rs =
  List.map (fun (_, k) -> Iarray.get prog.outs k) rs

let pp : type r. Format.formatter -> r prim -> unit =
 fun ppf op ->
  let (Operands xs) = operands op in
  Format.fprintf ppf "%s" (name op);
  let sep ppf () = Format.pp_print_string ppf "; " in
  (match op with
  | Map { prog; _ } ->
      let outs = Iarray.to_list prog.outs in
      let pr = printer prog outs in
      Format.fprintf ppf "%a [%a]" pr.bindings ()
        (Format.pp_print_list ~pp_sep:sep pr.expr)
        outs
  | Reduce { prog; axes; reductions; _ } ->
      let rs = reductions_list reductions in
      let pr = printer prog (reduced_roots prog rs) in
      Format.fprintf ppf "%a [%a] over %a" pr.bindings ()
        (Format.pp_print_list ~pp_sep:sep (pp_reduced prog pr))
        rs pp_shape axes
  | Scan { prog; axis; reduction = s; _ } ->
      let r = reduction s in
      let pr = printer prog (reduced_roots prog [ r ]) in
      Format.fprintf ppf "%a [%a] along %d" pr.bindings () (pp_reduced prog pr)
        r axis
  | Gather { axis; _ } -> Format.fprintf ppf " along %d" axis
  | Scatter { combine; unique; axis; _ } ->
      let combine =
        match (combine : Nx_kernel.combine) with
        | Set -> "set"
        | Add -> "add"
        | Max -> "max"
        | Min -> "min"
      in
      Format.fprintf ppf " %s%s along %d" combine
        (if unique then " unique" else "")
        axis
  | Sort { axis; descending; k; _ } ->
      Format.fprintf ppf " along %d%s%s" axis
        (if descending then " descending" else "")
        (match k with Some k -> Printf.sprintf " keeping %d" k | None -> "")
  | Fft f ->
      let kind =
        match f with
        | C2c { direction = Forward; _ } -> "C2c forward"
        | C2c { direction = Inverse; _ } -> "C2c inverse"
        | R2c _ -> "R2c"
        | C2r { n; _ } -> Printf.sprintf "C2r of %d points" n
      in
      Format.fprintf ppf " %s along %a" kind pp_shape (fft_axes f)
  | Linalg l -> Format.fprintf ppf " %s" (routine_name l)
  | Fold { shape; _ } -> Format.fprintf ppf " into %a" pp_shape shape
  | Assemble { shape; _ } -> Format.fprintf ppf " into %a" pp_shape shape
  | Contract _ | Copy _ | Move _ | Bitcast _ | Place _ | Check _ -> ());
  List.iteri (fun i x -> Format.fprintf ppf " (x%d: %a)" i pp_operand x) xs

(* Rules *)

let same_dtype a b = D.code a = D.code b
let load_placement (type d) (l : d load) : d Devices.placement option =
  let (Any x) = load_any l in
  at x

(* The shape [l] gives its loop, or its refusal raised naming [by]: a padded
   load's fill is an element of its dtype, and its pad fits its operand. *)
let load_shape (type d) ~by (l : d load) =
  match l with
  | Plain x -> shape x
  | Padded { x; fill; pad } -> (
      (match P.bits (dtype x) fill with
      | _ -> ()
      | exception Invalid_argument e -> invalid_argf "%s: %s" by e);
      match Shape.padded pad (shape x) with
      | Ok s -> s
      | Error e -> invalid_argf "%s: %s" by e)

(* Whether [x]'s shape is [l]'s, allocating nothing. *)
let has_layout_shape x l =
  let r = L.rank l in
  rank x = r
  &&
  let i = ref 0 in
  while !i < r && dim x !i = L.dim l !i do
    incr i
  done;
  !i = r

(* A loop's rule on its loads and coordinates: a load per operand of [prog],
   of its dtype and of [layout]'s shape, which is C-contiguous, and a
   coordinate along an axis of that shape. [what] names the loop. *)
let check_loads (type d) ~by ~what layout (prog : P.t) (loads : d load array)
    =
  let ins = prog.ins in
  if Array.length loads <> Iarray.length ins then
    invalid_argf "%s: a program of %d operands over %d loads" by
      (Iarray.length ins) (Array.length loads);
  Array.iteri
    (fun i l ->
      let (D.Any want) = Iarray.get ins i in
      let (D.Any have) = load_dtype l in
      if not (same_dtype want have) then
        invalid_argf "%s: load %d is %s where the program reads %s" by i
          (D.name have) (D.name want))
    loads;
  if not (L.is_contiguous layout && L.offset layout = 0) then
    invalid_argf "%s: a %s's layout %a is not C-contiguous" by what L.pp layout;
  Array.iteri
    (fun i l ->
      let fits =
        match l with
        | Plain x -> has_layout_shape x layout
        | Padded _ -> load_shape ~by l = L.shape layout
      in
      if not fits then
        invalid_argf "%s: load %d has shape %a, the %s %a" by i pp_shape
          (load_shape ~by l) what pp_shape (L.shape layout))
    loads;
  for i = 0 to Iarray.length prog.nodes - 1 do
    match Iarray.get prog.nodes i with
    | P.Coord a when a >= L.rank layout ->
        invalid_argf "%s: a coordinate along axis %d of a %s of rank %d" by a
          what (L.rank layout)
    | _ -> ()
  done

(* A map's rule; its result's layout, C-contiguous of [shape]. *)
let check_map (type d r) ~by layout (prog : P.t) (outs : (d, r) outs)
    (loads : d load array) =
  check_loads ~by ~what:"map" layout prog loads;
  let produced = prog.outs in
  let rec check : type q. int -> (d, q) outs -> unit =
   fun k -> function
     | [] ->
         if k <> Iarray.length produced then
           invalid_argf "%s: %d results for a program of %d outputs" by k
             (Iarray.length produced)
     | dt :: rest ->
         if k >= Iarray.length produced then
           invalid_argf "%s: more results than the program's %d outputs" by
             (Iarray.length produced);
         let (D.Any want) = P.dtype prog (Iarray.get produced k) in
         if not (same_dtype want dt) then
           invalid_argf "%s: result %d is %s where the program gives %s" by k
             (D.name dt) (D.name want);
         check (k + 1) rest
  in
  check 0 outs

(* A reduction's rule over [prog]'s outputs, reducing [axes] of [shape]: its
   output exists and is of a dtype it takes, and an extreme has a term. *)
let check_reduction ~by (prog : P.t) shape axes (r, k) =
  let produced = prog.outs in
  if k < 0 || k >= Iarray.length produced then
    invalid_argf "%s: a reduction of output %d of a program of %d" by k
      (Iarray.length produced);
  let (D.Any have) = P.dtype prog (Iarray.get produced k) in
  if not (Nx_kernel.accepts r have) then
    invalid_argf "%s: %s does not take %s" by (reduction_name r) (D.name have);
  let empty = Array.exists (fun a -> shape.(a) = 0) axes in
  match r with
  | (Monoid (Max | Min) | Arg _) when empty ->
      invalid_argf "%s: %s of no term" by (reduction_name r)
  | Monoid _ | Moments | Arg _ -> ()

(* [axes] strictly increasing, each an axis of [shape]. *)
let check_axes ~by shape axes =
  Array.iteri
    (fun i a ->
      if a < 0 || a >= Array.length shape then
        invalid_argf "%s: axis %d of a value of rank %d" by a
          (Array.length shape);
      if i > 0 && axes.(i - 1) >= a then
        invalid_argf "%s: axes %a are not strictly increasing" by pp_shape axes)
    axes

let reduction_width : type d a. (d, a) reduction -> int = function
  | Monoid _ -> 1
  | Moments _ | Arg _ -> 2

let rec results_width : type d r. (d, r) reductions -> int = function
  | [] -> 0
  | r :: rest -> reduction_width r + results_width rest

(* A loop's loads and results number at most [Prog.max_operands], as a
   program's operands and outputs do. *)
let check_width ~by loads results =
  if Array.length loads + results > P.max_operands then
    invalid_argf "%s: %d loads and %d results, past %d" by (Array.length loads)
      results P.max_operands

let check_reduce (type d r) ~by layout axes prog (rs : (d, r) reductions)
    (loads : d load array) =
  check_loads ~by ~what:"reduction" layout prog loads;
  let shape = L.shape layout in
  check_axes ~by shape axes;
  (match reductions_list rs with
  | [] -> invalid_argf "%s: a reduction of no output" by
  | l -> List.iter (check_reduction ~by prog shape axes) l);
  check_width ~by loads (results_width rs)

let check_scan (type d r) ~by layout axis prog (r : (d, r) reduction)
    (loads : d load array) =
  check_loads ~by ~what:"scan" layout prog loads;
  let shape = L.shape layout in
  check_axes ~by shape [| axis |];
  check_reduction ~by prog shape [||] (reduction r);
  check_width ~by loads (reduction_width r)

(* [shape] without [axes]. *)
let reduced shape axes =
  Array.of_list
    (List.filteri (fun a _ -> not (Array.mem a axes)) (Array.to_list shape))

(* Each operation's route: where it reads its operands, in [operands]'s order,
   and where its results lie. [results] and [prepare] take placements from it
   alone. *)

(* A map's route, its loads of the map's layout's shape by its rule. *)
(* A loop with a padded load reads its operands whole and computes the whole
   result on each device: a device's window of a padded, windowed read is no
   window of its operand. *)
let padded_route (type d) ~by (loads : d load array) =
  let operand l =
    let (Any x) = load_any l in
    shape x
  in
  Route.route ~by Replicated
    (Array.map load_placement loads)
    (Array.map operand loads)

let map_route (type d) ~by layout (loads : d load array) : d Route.t option =
  if Array.length loads = 0 then None
  else if Array.exists is_padded loads then padded_route ~by loads
  else
    let shape = L.shape layout in
    Route.route ~by Elementwise
      (Array.map load_placement loads)
      (Array.make (Array.length loads) shape)

let rec make_outs : type d r.
    ('a 'b 'c. int -> ('a, 'b, 'c) form -> ('a, 'b, 'c) t) ->
    int ->
    L.t ->
    d Devices.placement option ->
    (d, r) outs ->
    r =
 fun m k layout placement -> function
  | [] -> ()
  | dtype :: rest ->
      let v = m k (of_layout dtype layout placement) in
      (v, make_outs m (k + 1) layout placement rest)

(* A reduction's results, from position [k] of the operation's results. *)
let make_reduction : type d a.
    ('v 's 'c. int -> ('v, 's, 'c) form -> ('v, 's, 'c) t) ->
    int ->
    L.t ->
    d Devices.placement option ->
    (d, a) reduction ->
    a =
 fun m k layout placement -> function
  | Monoid (_, _, dtype) -> m k (of_layout dtype layout placement)
  | Moments (_, dtype) ->
      let mean = m k (of_layout dtype layout placement) in
      (mean, m (k + 1) (of_layout dtype layout placement))
  | Arg (_, _, dtype) ->
      let extreme = m k (of_layout dtype layout placement) in
      (extreme, m (k + 1) (of_layout D.Int64 layout placement))

let rec make_reductions : type d r.
    ('v 's 'c. int -> ('v, 's, 'c) form -> ('v, 's, 'c) t) ->
    int ->
    L.t ->
    d Devices.placement option ->
    (d, r) reductions ->
    r =
 fun m k layout placement -> function
  | [] -> ()
  | r :: rest ->
      let v = make_reduction m k layout placement r in
      (v, make_reductions m (k + reduction_width r) layout placement rest)

(* A loop's route by [rule], its loads of [layout]'s shape. *)
let loop_route (type d) ~by rule layout (loads : d load array) :
    d Route.t option =
  if Array.length loads = 0 then None
  else if Array.exists is_padded loads then padded_route ~by loads
  else
    let shape = L.shape layout in
    Route.route ~by rule
      (Array.map load_placement loads)
      (Array.make (Array.length loads) shape)

let one_route ~by rule x = Route.route ~by rule [| at x |] [| shape x |]

(* Where an operation of one operand by [rule], not [Replicated], reads [x] and
   puts its result, where [x] is of every set or lies at a placement that cuts
   no axis: where [x] lies. Elsewhere, where its route says. *)
let lies_simply (type v s d) (x : (v, s, d) t) =
  let simply p = not (Grid.is_cut (Devices.grid p)) in
  match x with
  | Array { at; _ } | Shards { at; _ } | Donated { at; _ } -> simply at
  | Deferred { form; _ } | Traced { form; _ } -> (
      match form.placement with None -> true | Some p -> simply p)

let one_result ~by rule x =
  if lies_simply x then at x
  else Option.map (fun (r : _ Route.t) -> r.result) (one_route ~by rule x)

(* [x]'s layout, as its form has it, allocating nothing for a value on one
   device. *)
let form_layout (type v s d) (x : (v, s, d) t) =
  match x with
  | Array { a; _ } -> Nx_array.layout a
  | Donated { arrays; _ } when Iarray.length arrays = 1 ->
      Nx_array.layout (Iarray.get arrays 0)
  | Shards _ | Donated _ -> L.contiguous (shape x)
  | Deferred { form; _ } | Traced { form; _ } -> form.layout

let result (r : _ Route.t option) =
  Option.map (fun (r : _ Route.t) -> r.result) r

(* [x]'s shape moved by [mv]. Raises naming [by] where [mv] does not apply. *)
let moved_shape ~by mv x =
  match Nx_array.Move.shape mv (shape x) with
  | s' -> s'
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e

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

(* Gathers, scatters and assemblies *)

(* Whether shapes [a] and [b] have one extent along every axis but [axis]. *)
let off_axis axis a b =
  Array.length a = Array.length b
  &&
  let ok = ref true in
  Array.iteri (fun i e -> if i <> axis && e <> b.(i) then ok := false) a;
  !ok

let check_axis ~by axis r =
  if axis < 0 || axis >= r then
    invalid_argf "%s: axis %d of an operand of rank %d" by axis r

let check_gather ~by axis idx x =
  let si = shape idx and sx = shape x in
  check_axis ~by axis (Array.length sx);
  if not (off_axis axis si sx) then
    invalid_argf "%s: positions %a do not fit an operand %a along axis %d" by
      pp_shape si pp_shape sx axis

let gather_route ~by axis idx x =
  Route.route ~by (Gather axis) [| at idx; at x |] [| shape idx; shape x |]

let check_scatter ~by (combine : Nx_kernel.combine) axis idx updates into =
  let si = shape idx and su = shape updates and st = shape into in
  check_axis ~by axis (Array.length st);
  if si <> su then
    invalid_argf "%s: positions %a and updates %a differ" by pp_shape si
      pp_shape su;
  if not (off_axis axis su st) then
    invalid_argf "%s: updates %a do not fit a target %a along axis %d" by
      pp_shape su pp_shape st axis;
  if combine = Add && not (P.accepts2 (Binary Add) (dtype into)) then
    invalid_argf "%s: Add does not take %s" by (D.name (dtype into))

let scatter_route ~by axis idx updates into =
  Route.route ~by (Into axis)
    [| at idx; at updates; at into |]
    [| shape idx; shape updates; shape into |]

(* A sort's values' shape, and its positions': its rule. *)
let sort_shape ~by axis k x =
  match Shape.sort ~axis ~k (shape x) with
  | Ok s -> s
  | Error e -> invalid_argf "%s: %s" by e

let sort_route ~by axis x =
  Route.route ~by (Along [| axis |]) [| at x |] [| shape x |]

(* [r]'s value, or its error raised naming [by]. *)
let shaped ~by = function Ok s -> s | Error e -> invalid_argf "%s: %s" by e

let route_along ~by axes (xs : _ any list) =
  let at_ = Array.of_list (List.map (fun (Any x) -> at x) xs) in
  let shapes = Array.of_list (List.map (fun (Any x) -> shape x) xs) in
  Route.route ~by (Along axes) at_ shapes

let check_fft (type d r) ~by (f : (d, r) fft) =
  let into x y fits =
    if not fits then
      invalid_argf "%s: a transform of %s into %s" by (D.name x) (D.name y)
  in
  match f with
  | C2c _ -> ()
  | R2c { dtype = c; x; _ } ->
      let r = dtype x in
      into r c (Nx_kernel.same_precision r c)
  | C2r { dtype = r; x; _ } ->
      let c = dtype x in
      into c r (Nx_kernel.same_precision r c)

(* A transform's result's shape: its rule. *)
let fft_shape (type d r) ~by (f : (d, r) fft) =
  check_fft ~by f;
  shaped ~by
    (match f with
    | C2c { axes; x; _ } -> Shape.c2c ~axes (shape x)
    | R2c { axes; x; _ } -> Shape.r2c ~axes (shape x)
    | C2r { axes; n; x; _ } -> Shape.c2r ~axes ~n (shape x))

let fft_route ~by f = route_along ~by (fft_axes f) [ fft_operand f ]

(* A routine's results' shapes, in their order: its rule. *)
let linalg_shapes (type d r) ~by (l : (d, r) linalg) =
  List.iter
    (fun (Any x) ->
      if not (Nx_kernel.full_precision (dtype x)) then
        invalid_argf "%s: %s does not take %s" by (routine_name l)
          (D.name (dtype x)))
    (linalg_operands l);
  let one r = Result.map (fun s -> [| s |]) r in
  let two r = Result.map (fun (s, s') -> [| s; s' |]) r in
  let three r = Result.map (fun (s, s', s'') -> [| s; s'; s'' |]) r in
  shaped ~by
    (match l with
    | Cholesky { a; _ } -> one (Shape.cholesky (shape a))
    | Lu a -> three (Shape.lu (shape a))
    | Qr { factors; a } -> two (Shape.qr factors (shape a))
    | Svd { factors; a } -> three (Shape.svd factors (shape a))
    | Svd_values a -> one (Shape.svd_values (shape a))
    | Eigh a -> two (Shape.eigh (shape a))
    | Eigh_values a -> one (Shape.eigh_values (shape a))
    | Eig a -> two (Shape.eig (shape a))
    | Eig_values a -> one (Shape.eig_values (shape a))
    | Solve_triangular { a; b; _ } ->
        one (Shape.solve_triangular (shape a) (shape b)))

(* A routine acts along its matrices' last two axes. *)
let linalg_route ~by l =
  let xs = linalg_operands l in
  let (Any a) = List.hd xs in
  let r = rank a in
  route_along ~by [| r - 2; r - 1 |] xs

(* A fold's rule: [whole] is a shape a layout admits, and [x] has the shape a
   padded load of it gives, and a dtype {!Nx_kernel.Prog.Add} takes. *)
let check_fold ~by whole pad x =
  if not (P.accepts2 (Binary Add) (dtype x)) then
    invalid_argf "%s: Fold does not take %s" by (D.name (dtype x));
  (match L.contiguous whole with
  | _ -> ()
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e);
  let s = shaped ~by (Shape.padded pad whole) in
  if not (has_shape x s) then
    invalid_argf "%s: an operand of shape %a, not %a" by pp_shape (shape x)
      pp_shape s

let fold_route ~by x = Route.route ~by Replicated [| at x |] [| shape x |]

let check_assemble (type v s d) ~by (dt : (v, s) dtype) whole (fill : v)
    (pieces : (Nx_array.Move.range array * (v, s, d) t) list) =
  (match L.contiguous whole with
  | _ -> ()
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e);
  (match P.bits dt fill with
  | _ -> ()
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e);
  List.iteri
    (fun i (rs, x) ->
      match Nx_array.Move.shape (Slice rs) whole with
      | s ->
          if not (has_shape x s) then
            invalid_argf "%s: piece %d has shape %a, its region %a" by i
              pp_shape (shape x) pp_shape s
      | exception Invalid_argument e -> invalid_argf "%s: piece %d: %s" by i e)
    pieces

(* An assembly reads each piece whole and lies whole on every device of their
   set: no piece's window is any device's alone. *)
let assemble_route (type v s d) ~by
    (pieces : (Nx_array.Move.range array * (v, s, d) t) list) : d Route.t option
    =
  match pieces with
  | [] -> None
  | _ ->
      Route.route ~by Replicated
        (Array.of_list (List.map (fun (_, x) -> at x) pieces))
        (Array.of_list (List.map (fun (_, x) -> shape x) pieces))

(* A contraction's operand shapes, [a], [b], then [init] where there is one. *)
let contract_shapes a b init =
  let s = [| shape a; shape b |] in
  match init with None -> s | Some i -> Array.append s [| shape i |]

let contract_shape ~by ~batch ~contracting ~acc a b init =
  (let (D.Any dt) = acc in
   if not (Nx_kernel.accumulates dt) then
     invalid_argf "%s: an accumulator of %s" by (D.name dt));
  let s = shaped ~by (Shape.contract ~batch ~contracting (shape a) (shape b)) in
  (match init with
  | Some i when not (has_shape i s) ->
      invalid_argf "%s: init of shape %a, not %a" by pp_shape (shape i) pp_shape
        s
  | Some _ | None -> ());
  s

(* A contraction reads its operands whole on each device that computes it: where
   they lie at one placement that cuts no axis, there; else on every device of
   their set, each holding them whole. *)
let contract_route (type a b c e v s d) ~by (a : (a, b, d) t)
    (b : (c, e, d) t) (init : (v, s, d) t option) : d Route.t option =
  let ps =
    match init with
    | None -> [| at a; at b |]
    | Some i -> [| at a; at b; at i |]
  in
  match Route.common ps with
  | Route.Every_set -> None
  | Route.Uncut p ->
      Some { Route.operands = Array.make (Array.length ps) p; result = p }
  | Route.Other -> Route.route ~by Replicated ps (contract_shapes a b init)

let linalg_results : type d r.
    by:string ->
    ('v 's 'c. int -> ('v, 's, 'c) form -> ('v, 's, 'c) t) ->
    (d, r) linalg ->
    r =
 fun ~by m l ->
  let shapes = linalg_shapes ~by l in
  let placement = result (linalg_route ~by l) in
  let one : type v s. int -> (v, s) dtype -> (v, s, d) t =
   fun k dtype -> m k (of_layout dtype (L.contiguous shapes.(k)) placement)
  in
  (* Results are made in position order. *)
  let two a b =
    let x = one 0 a in
    (x, one 1 b)
  in
  let three a b c =
    let x = one 0 a in
    let y = one 1 b in
    (x, y, one 2 c)
  in
  match l with
  | Cholesky { a; _ } -> one 0 (dtype a)
  | Lu a -> three (dtype a) D.Int64 D.Int64
  | Qr { a; _ } -> two (dtype a) (dtype a)
  | Svd { a; _ } -> three (dtype a) (dtype a) (dtype a)
  | Svd_values a -> one 0 (dtype a)
  | Eigh a -> two (dtype a) (dtype a)
  | Eigh_values a -> one 0 (dtype a)
  | Eig a -> two (dtype a) (dtype a)
  | Eig_values a -> one 0 (dtype a)
  | Solve_triangular { a; _ } -> one 0 (dtype a)

let results : type r.
    by:string ->
    ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
    r prim ->
    r =
 fun ~by m op ->
  match op with
  | Map { layout; prog; outs; loads } ->
      check_map ~by layout prog outs loads;
      let placement = result (map_route ~by layout loads) in
      make_outs m 0 layout placement outs
  | Reduce { layout; axes; prog; reductions; loads } ->
      check_reduce ~by layout axes prog reductions loads;
      let placement = result (loop_route ~by (Reduce axes) layout loads) in
      let out = L.contiguous (reduced (L.shape layout) axes) in
      make_reductions m 0 out placement reductions
  | Scan { layout; axis; prog; reduction; loads } ->
      check_scan ~by layout axis prog reduction loads;
      let placement = result (loop_route ~by (Along [| axis |]) layout loads) in
      make_reduction m 0 layout placement reduction
  | Gather { axis; idx; x } ->
      check_gather ~by axis idx x;
      let placement = result (gather_route ~by axis idx x) in
      m 0 (of_layout (dtype x) (L.contiguous (shape idx)) placement)
  | Scatter { combine; axis; idx; updates; into; _ } ->
      check_scatter ~by combine axis idx updates into;
      let placement = result (scatter_route ~by axis idx updates into) in
      m 0 (of_layout (dtype into) (L.contiguous (shape into)) placement)
  | Sort { axis; k; x; _ } ->
      let s = sort_shape ~by axis k x in
      let placement = result (sort_route ~by axis x) in
      let layout = L.contiguous s in
      let values = m 0 (of_layout (dtype x) layout placement) in
      (values, m 1 (of_layout D.Int64 layout placement))
  | Fft f ->
      let layout = L.contiguous (fft_shape ~by f) in
      let placement = result (fft_route ~by f) in
      (match f with
      | C2c { x; _ } -> m 0 (of_layout (dtype x) layout placement)
      | R2c { dtype; _ } -> m 0 (of_layout dtype layout placement)
      | C2r { dtype; _ } -> m 0 (of_layout dtype layout placement))
  | Linalg l -> linalg_results ~by m l
  | Fold { shape; pad; x } ->
      check_fold ~by shape pad x;
      let placement = result (fold_route ~by x) in
      m 0 (of_layout (dtype x) (L.contiguous shape) placement)
  | Assemble { dtype; shape; fill; pieces } ->
      check_assemble ~by dtype shape fill pieces;
      let placement = result (assemble_route ~by pieces) in
      m 0 (of_layout dtype (L.contiguous shape) placement)
  | Contract { batch; contracting; acc; out; a; b; init } ->
      let shape = contract_shape ~by ~batch ~contracting ~acc a b init in
      let placement = result (contract_route ~by a b init) in
      m 0 (of_layout out (L.contiguous shape) placement)
  | Copy x ->
      let placement = one_result ~by Elementwise x in
      m 0 (of_layout (dtype x) (L.contiguous (shape x)) placement)
  | Move (mv, x) ->
      let s' = moved_shape ~by mv x in
      let placement = one_result ~by (Move mv) x in
      let layout =
        match L.move mv (form_layout x) with
        | Some l -> l
        | None -> L.contiguous s'
      in
      m 0 (of_layout (dtype x) layout placement)
  | Bitcast (dt, x) ->
      let layout =
        bitcast_layout ~by (form_layout x) (D.Any (dtype x)) (D.Any dt)
      in
      let placement = one_result ~by (bitcast_rule dt x) x in
      m 0 (of_layout dt layout placement)
  | Place (p, x) ->
      let s = shape x in
      ignore (Devices.window ~by p s 0);
      let layout =
        match (x, Devices.device p) with
        | Array { a; _ }, Some _ -> Nx_array.layout a
        | _ -> L.contiguous s
      in
      m 0 (of_layout (dtype x) layout (Some p))
  | Check { ok; data; _ } ->
      let s = shape ok in
      List.iteri
        (fun i (Any x) ->
          if shape x <> s then
            invalid_argf "%s: datum %d has shape %a, the flag %a" by i pp_shape
              (shape x) pp_shape s)
        data

(* One-node maps *)

(* Raises naming [by]: [node] does not take operands of [ins]. *)
let refused ~by node ins =
  invalid_argf "%s: %s does not take %s" by (kind node)
    (String.concat ", "
       (List.map (fun (D.Any dt) -> D.name dt) (Iarray.to_list ins)))

let one_node node dt ins loads =
  let (Any x) = load_any loads.(0) in
  let shape = shape x in
  Map
    {
      layout = L.contiguous shape;
      prog = P.of_node ~ins node;
      outs = Value.[ dt ];
      loads;
    }

let op1 ~by k dt x =
  let node = P.Op1 (k, D.Any dt, 0) in
  let ins : D.any iarray = [| D.Any (dtype x) |] in
  if not (P.accepts1 k (dtype x) dt) then refused ~by node ins;
  one_node node dt ins [| Plain x |]

let op2 ~by k dt x y =
  let i = D.Any (dtype x) in
  let node = P.Op2 (k, 0, 1) in
  let ins : D.any iarray = [| i; i |] in
  if not (P.accepts2 k (dtype x)) then refused ~by node ins;
  one_node node dt ins [| Plain x; Plain y |]

let op3 ~by k c x y =
  let i = D.Any (dtype x) in
  let node = P.Op3 (k, 0, 1, 2) in
  let ins : D.any iarray = [| D.Any (dtype c); i; i |] in
  if not (P.accepts3 k (dtype c) (dtype x)) then refused ~by node ins;
  one_node node (dtype x) ins [| Plain c; Plain x; Plain y |]

(* Where a route reads operand [i]: [None] where every operand is of every
   set. *)
let read_at (r : _ Route.t option) i =
  Option.map (fun (r : _ Route.t) -> r.operands.(i)) r

(* A loop's loads placed by [place] where its route [r] reads them; [None]
   where none moves. *)
let placed : type d.
    ('v 's 'e. 'e Devices.placement option -> ('v, 's, 'e) t -> ('v, 's, 'e) t) ->
    d Route.t option ->
    d load array ->
    d load array option =
 fun place r loads ->
  let moved = ref false in
  let loads =
    Array.mapi
      (fun i l ->
        match l with
        | Plain x ->
            let y = place (read_at r i) x in
            if y == x then l
            else begin
              moved := true;
              Plain y
            end
        | Padded p ->
            let y = place (read_at r i) p.x in
            if y == p.x then l
            else begin
              moved := true;
              Padded { p with x = y }
            end)
      loads
  in
  if !moved then Some loads else None

(* [x] placed by [place] where an operation of one operand by [rule] reads
   it. *)
let placed_one : type v s d.
    by:string ->
    ('w 'r 'e. 'e Devices.placement option -> ('w, 'r, 'e) t -> ('w, 'r, 'e) t) ->
    Route.rule ->
    (v, s, d) t ->
    (v, s, d) t =
 fun ~by place rule x ->
  if lies_simply x then place (at x) x
  else place (read_at (one_route ~by rule x) 0) x

let prepare : type r.
    by:string ->
    ('v 's 'd. 'd Devices.placement option -> ('v, 's, 'd) t -> ('v, 's, 'd) t) ->
    r prim ->
    r prim =
 fun ~by place op ->
  match op with
  | Map p -> (
      check_map ~by p.layout p.prog p.outs p.loads;
      match placed place (map_route ~by p.layout p.loads) p.loads with
      | Some loads -> Map { p with loads }
      | None -> op)
  | Reduce p -> (
      check_reduce ~by p.layout p.axes p.prog p.reductions p.loads;
      let r = loop_route ~by (Reduce p.axes) p.layout p.loads in
      match placed place r p.loads with
      | Some loads -> Reduce { p with loads }
      | None -> op)
  | Scan p -> (
      check_scan ~by p.layout p.axis p.prog p.reduction p.loads;
      let r = loop_route ~by (Along [| p.axis |]) p.layout p.loads in
      match placed place r p.loads with
      | Some loads -> Scan { p with loads }
      | None -> op)
  | Gather g ->
      check_gather ~by g.axis g.idx g.x;
      let r = gather_route ~by g.axis g.idx g.x in
      Gather
        { g with idx = place (read_at r 0) g.idx; x = place (read_at r 1) g.x }
  | Scatter s ->
      check_scatter ~by s.combine s.axis s.idx s.updates s.into;
      let r = scatter_route ~by s.axis s.idx s.updates s.into in
      Scatter
        {
          s with
          idx = place (read_at r 0) s.idx;
          updates = place (read_at r 1) s.updates;
          into = place (read_at r 2) s.into;
        }
  | Sort s ->
      ignore (sort_shape ~by s.axis s.k s.x);
      let r = sort_route ~by s.axis s.x in
      Sort { s with x = place (read_at r 0) s.x }
  | Fold f ->
      check_fold ~by f.shape f.pad f.x;
      Fold { f with x = place (read_at (fold_route ~by f.x) 0) f.x }
  | Fft f ->
      ignore (fft_shape ~by f);
      let r = fft_route ~by f in
      Fft (map_fft (fun x -> place (read_at r 0) x) f)
  | Linalg l ->
      ignore (linalg_shapes ~by l);
      let r = linalg_route ~by l in
      (* CR: Pass operand indices through map_linalg and remove this counter.
         Record fields do not guarantee a-then-b callbacks: reversed calls
         gather batch-split b[2;2;1] onto both devices and slice replicated
         a[2;2;2]. Use a=0, b=1, and 0 for unary routines;
         Prim.map can ignore the index. Preserve the route's views. *)
      (* The operands in order: [a], then [b]. *)
      let i = ref (-1) in
      Linalg
        (map_linalg
           (fun x ->
             incr i;
             place (read_at r !i) x)
           l)
  | Assemble a ->
      check_assemble ~by a.dtype a.shape a.fill a.pieces;
      let r = assemble_route ~by a.pieces in
      Assemble
        {
          a with
          pieces =
            List.mapi (fun i (rs, x) -> (rs, place (read_at r i) x)) a.pieces;
        }
  | Contract c ->
      ignore
        (contract_shape ~by ~batch:c.batch ~contracting:c.contracting
           ~acc:c.acc c.a c.b c.init);
      let r = contract_route ~by c.a c.b c.init in
      let a = place (read_at r 0) c.a and b = place (read_at r 1) c.b in
      let init = Option.map (fun i -> place (read_at r 2) i) c.init in
      if a == c.a && b == c.b && init == c.init then op
      else Contract { c with a; b; init }
  | Copy x -> Copy (placed_one ~by place Elementwise x)
  | Move (mv, x) ->
      ignore (moved_shape ~by mv x);
      Move (mv, placed_one ~by place (Move mv) x)
  | Bitcast (dt, x) -> Bitcast (dt, placed_one ~by place (bitcast_rule dt x) x)
  | Place _ | Check _ -> op

let rec iter_outs : type d q.
    ('v 's. ('v, 's, d) t -> unit) -> (d, q) outs -> q -> unit =
 fun f outs r ->
  match (outs, r) with
  | [], () -> ()
  | _ :: rest, (v, r) ->
      f v;
      iter_outs f rest r

let iter_reduction : type d a.
    ('v 's. ('v, 's, d) t -> unit) -> (d, a) reduction -> a -> unit =
 fun f r v ->
  match (r, v) with
  | Monoid _, v -> f v
  | Moments _, (a, b) ->
      f a;
      f b
  | Arg _, (a, b) ->
      f a;
      f b

let rec iter_reductions : type d q.
    ('v 's. ('v, 's, d) t -> unit) -> (d, q) reductions -> q -> unit =
 fun f rs v ->
  match (rs, v) with
  | [], () -> ()
  | r :: rest, (a, v) ->
      iter_reduction f r a;
      iter_reductions f rest v

let iter_fft : type d r.
    ('v 's. ('v, 's, d) t -> unit) -> (d, r) fft -> r -> unit =
 fun f t r ->
  match (t, r) with C2c _, x -> f x | R2c _, x -> f x | C2r _, x -> f x

let iter_linalg : type d r.
    ('v 's. ('v, 's, d) t -> unit) -> (d, r) linalg -> r -> unit =
 fun f l r ->
  match (l, r) with
  | Cholesky _, x -> f x
  | Svd_values _, x -> f x
  | Eigh_values _, x -> f x
  | Eig_values _, x -> f x
  | Solve_triangular _, x -> f x
  | Lu _, (a, b, c) ->
      f a;
      f b;
      f c
  | Svd _, (a, b, c) ->
      f a;
      f b;
      f c
  | Qr _, (a, b) ->
      f a;
      f b
  | Eigh _, (a, b) ->
      f a;
      f b
  | Eig _, (a, b) ->
      f a;
      f b

let iter_results : type r.
    ('v 's 'd. ('v, 's, 'd) t -> unit) -> r prim -> r -> unit =
 fun f op r ->
  match op with
  | Map { outs; _ } -> iter_outs f outs r
  | Reduce { reductions; _ } -> iter_reductions f reductions r
  | Scan { reduction; _ } -> iter_reduction f reduction r
  | Sort _ ->
      let values, positions = r in
      f values;
      f positions
  | Fft t -> iter_fft f t r
  | Linalg l -> iter_linalg f l r
  | Gather _ -> f r
  | Scatter _ -> f r
  | Fold _ -> f r
  | Assemble _ -> f r
  | Contract _ -> f r
  | Copy _ -> f r
  | Move _ -> f r
  | Bitcast _ -> f r
  | Place _ -> f r
  | Check _ -> ()

let any xs =
  Array.init (Iarray.length xs) (fun j -> Nx_array.Any (Iarray.get xs j))

(* [x]'s arrays untyped; a value on one device builds no [iarray] on the way. *)
let any_arrays : type v s d. (v, s, d) t -> Nx_array.any array = function
  | Array { a; _ } -> [| Nx_array.Any a |]
  | x -> any (arrays x)

let result_arrays op r =
  let acc = ref ([] : Nx_array.any array list) in
  iter_results (fun x -> acc := List.cons (any_arrays x) !acc) op r;
  Array.of_list (List.rev !acc)
