open Tolk

(* An element carries the memory it views, if any, so that a store through a
   movement of storage knows where it writes. Call-local storage numbers its
   slots apart from parameters and buffers. *)
type place = { scratch : bool; slot : int; index : int }
type cell = { value : Dtype.const; at : place option }
type tensor = { shape : int list; cells : cell array }

let fail fmt = Format.kasprintf invalid_arg fmt

let concrete u =
  List.map
    (function
      | Ops.Int n -> n
      | Sym _ -> fail "cannot evaluate a %a of symbolic shape" Op.pp (Ops.op u))
    (Shape.shape u)

let size shape = List.fold_left ( * ) 1 shape

let count r = Bigint.to_int (Shape.to_z (Ops.nth r 0))

let int = function
  | Ops.Int n -> n
  | Sym u -> (
      match Shape.ssimplify u with
      | Int n -> n
      | Sym _ -> fail "a movement is symbolic")

(* Row-major coordinates *)

let coords shape k =
  List.fold_right (fun n (k, acc) -> (k / n, (k mod n) :: acc)) shape (k, [])
  |> snd

let offset shape idx =
  List.fold_left2 (fun acc n i -> (acc * n) + i) 0 shape idx

(* [gather shape t f] is the tensor of [shape] whose element at [idx] is [t]'s
   at [f idx], or [fill] where [f idx] is [None]. *)
let gather ?fill shape t f =
  let cell k =
    match (f (coords shape k), fill) with
    | Some idx, _ -> t.cells.(offset t.shape idx)
    | None, Some value -> { value; at = None }
    | None, None -> fail "an element has no source"
  in
  { shape; cells = Array.init (size shape) cell }

(* [index u t idxs] is the gather [u] of [t]: each index tensor of [idxs] reads
   one leading axis of [t] at the position it holds, where its own axes are the
   result's, and the rest of [t] follows. An element at an Invalid index is
   Invalid and views no memory: a store through it is dropped. *)
let index u t idxs =
  let position c =
    match c.value with
    | `Int n -> Some (Bigint.to_int n)
    | `Invalid -> None
    | v -> fail "a gather's index is %a" Dtype.pp_const v
  in
  gather ~fill:`Invalid (concrete u) t (fun c ->
      let rec read c = function
        | [] -> Some c
        | it :: rest -> (
            let k = List.length it.shape in
            let here = List.filteri (fun i _ -> i < k) c in
            match position it.cells.(offset it.shape here) with
            | None -> None
            | Some i ->
                Option.map (List.cons i)
                  (read (List.filteri (fun i _ -> i >= k) c) rest))
      in
      match read c idxs with
      | None -> None
      | Some at when List.for_all2 (fun i n -> 0 <= i && i < n) at t.shape ->
          Some at
      | Some _ -> fail "a gather reads outside its source")

let broadcast shape t =
  if shape = t.shape then t
  else
    let lead = List.length shape - List.length t.shape in
    gather shape t (fun idx ->
        Some
          (List.map2
             (fun n i -> if n = 1 then 0 else i)
             t.shape
             (List.filteri (fun a _ -> a >= lead) idx)))

let movement u t =
  let shape = concrete u in
  let zero = Dtype.const (Ops.dtype u) (`Int Bigint.zero) in
  match Shape.marg u with
  | Reshape _ -> { t with shape }
  | Expand _ -> broadcast shape t
  | Pad pads ->
      gather ~fill:zero shape t (fun idx ->
          let src = List.map2 (fun (start, _) i -> i - int start) pads idx in
          if List.for_all2 (fun n i -> 0 <= i && i < n) t.shape src then
            Some src
          else None)
  | Shrink box ->
      gather shape t (fun idx ->
          Some (List.map2 (fun (start, _) i -> i + int start) box idx))
  | Permute order ->
      gather shape t (fun idx ->
          Some
            (List.init (List.length order) (fun a ->
                 let rec find b = function
                   | o :: _ when o = a -> List.nth idx b
                   | _ :: rest -> find (b + 1) rest
                   | [] -> fail "a permutation misses axis %d" a
                 in
                 find 0 order)))

(* Elements *)

let held dt : Dtype.const -> Dtype.const = function
  | #Dtype.value as v -> (
      match Dtype.const dt v with
      | #Dtype.value as v -> (Dtype.truncate dt v :> Dtype.const)
      | `Invalid -> `Invalid)
  | `Invalid -> `Invalid

let is_invalid : Dtype.const -> bool = function
  | `Invalid -> true
  | #Dtype.value -> false

let element u (values : Dtype.const list) : Dtype.const =
  let dt = Ops.dtype u in
  match (Ops.op u, values) with
  | Where, `Invalid :: _ -> `Invalid
  | Where, [ `Bool c; a; b ] -> if c then a else b
  | _, values when List.exists is_invalid values -> `Invalid
  | Cast, [ v ] -> held dt v
  | Bitcast, [ (#Dtype.value as v) ] ->
      (Dtype.bitcast (Ops.dtype (Ops.nth u 0)) dt v :> Dtype.const)
  | op, values when Op.Set.mem op Op.Set.alu ->
      held dt (Ops.exec_alu ~truncate_output:false op dt values)
  | op, _ -> fail "cannot evaluate a %a" Op.pp op

let elementwise u ts =
  let shape = concrete u in
  let ts = List.map (broadcast shape) ts in
  let cell k =
    { value = element u (List.map (fun t -> t.cells.(k).value) ts); at = None }
  in
  { shape; cells = Array.init (size shape) cell }

let fold op dt (values : Dtype.const list) =
  List.fold_left
    (fun acc v ->
      held dt (Ops.exec_alu ~truncate_output:false op dt [ acc; v ]))
    (Ops.identity_element op dt)
    values

let reduce u op axes t =
  let shape = concrete u in
  let kept = List.filteri (fun a _ -> a >= axes) t.shape in
  let n = size (List.filteri (fun a _ -> a < axes) t.shape) in
  let cell k =
    let values = List.init n (fun j -> t.cells.((j * size kept) + k).value) in
    { value = fold op (Ops.dtype u) values; at = None }
  in
  { shape; cells = Array.init (size shape) cell }

let stack u ts =
  let shape = concrete u in
  let tail = List.tl shape in
  let cells = List.map (fun t -> (broadcast tail t).cells) ts in
  { shape; cells = Array.concat cells }

(* Bit reinterpretation between types of different sizes: the bytes of each row
   of the last axis, little-endian, read as the other type. *)

let unsigned dt =
  match Dtype.itemsize dt with
  | 1 -> Dtype.Uint8
  | 2 -> Uint16
  | 4 -> Uint32
  | 8 -> Uint64
  | n -> fail "no unsigned type of %d bytes" n

let bits dt : Dtype.const -> Bigint.t option = function
  | `Invalid -> None
  | #Dtype.value as v -> (
      match Dtype.bitcast dt (unsigned dt) v with
      | `Int z -> Some z
      | _ -> fail "the bits of a %a" Dtype.pp dt)

let rebytes u t =
  let from = Ops.dtype (Ops.nth u 0) and dt = Ops.dtype u in
  let os = Dtype.itemsize from and ns = Dtype.itemsize dt in
  let shape = concrete u in
  let last s = List.nth s (List.length s - 1) in
  let row_in = last t.shape and row_out = last shape in
  let cell k =
    let row = k / row_out and at = k mod row_out in
    let byte b =
      let e = t.cells.((row * row_in) + (b / os)) in
      let shift = 8 * (b mod os) in
      Option.map
        (fun z ->
          Bigint.logand (Bigint.shift_right z shift) (Bigint.of_int 0xff))
        (bits from e.value)
    in
    let rec gather i acc =
      if i = ns then Some acc
      else
        match byte ((at * ns) + i) with
        | Some v ->
            gather (i + 1) (Bigint.logor acc (Bigint.shift_left v (8 * i)))
        | None -> None
    in
    let value : Dtype.const =
      match gather 0 Bigint.zero with
      | Some z -> (Dtype.bitcast (unsigned dt) dt (`Int z) :> Dtype.const)
      | None -> `Invalid
    in
    { value; at = None }
  in
  { shape; cells = Array.init (size shape) cell }

let fresh t =
  { t with cells = Array.map (fun c -> { c with at = None }) t.cells }

(* Memory *)

(* What stores wrote, by place: scratch or not, slot and index. *)
module Written = Hashtbl.Make (struct
  type t = bool * int * int

  let equal (s0, l0, i0) (s1, l1, i1) =
    Bool.equal s0 s1 && Int.equal l0 l1 && Int.equal i0 i1

  let hash (s, l, i) = (((i * 65599) + l) * 2) + Bool.to_int s
end)

type memory = {
  buffers : (int * Dtype.value array) list;
  written : Dtype.const Written.t;
}

let initial m at : Dtype.const =
  match List.assoc_opt at.slot m.buffers with
  | _ when at.scratch -> `Invalid
  | None -> `Invalid
  | Some a when at.index < Array.length a -> (a.(at.index) :> Dtype.const)
  | Some _ -> fail "element %d is outside memory %d" at.index at.slot

let key at = (at.scratch, at.slot, at.index)

let current m at =
  match Written.find_opt m.written (key at) with
  | Some v -> v
  | None -> initial m at

let storage m u =
  match Ops.arg u with
  | Param { slot; addrspace; _ } when addrspace <> Alu ->
      let cell index =
        let at = { scratch = Ops.op u = Alloc; slot; index } in
        { value = initial m at; at = Some at }
      in
      { shape = concrete u; cells = Array.init (Shape.max_numel u) cell }
  | _ -> fail "cannot evaluate a variable"

let store m dst value =
  let value = broadcast dst.shape value in
  Array.iteri
    (fun k c ->
      match (c.at, c.value) with
      | Some at, _ -> Written.replace m.written (key at) value.cells.(k).value
      | None, `Invalid -> ()
      | None, _ -> fail "a store's destination is not storage")
    dst.cells

let reread m t =
  let cell c =
    match c.at with Some at -> { c with value = current m at } | None -> c
  in
  { t with cells = Array.map cell t.cells }

(* The value of an effect, which nothing reads. *)
let none = { shape = []; cells = [||] }

(* [run m params u] is the value of [u], each parameter of slot [k] standing for
   the value [params] gives [k], if any. *)
let rec run m params u =
  let values = Ops.Tbl.create 64 in
  (* The storage each loop writes: that of the Afters that list it. *)
  let loops = Ops.Tbl.create 8 in
  List.iter
    (fun a ->
      if Ops.op a = After then
        List.iter
          (fun e ->
            if Ops.op e = End then
              Ops.Tbl.replace loops e
                (Ops.storage_base a
                :: Option.value (Ops.Tbl.find_opt loops e) ~default:[]))
          (List.tl (Ops.src a)))
    (Ops.toposort ~calls:Skip u);
  let rec value u =
    match Ops.Tbl.find_opt values u with
    | Some v -> v
    | None ->
        let v = compute u in
        Ops.Tbl.replace values u v;
        v
  (* [call read c] runs the body of the call [c] on its arguments, as [read]
     gives them. *)
  and call read c =
    let args = List.mapi (fun k a -> (k, read a)) (Ops.src_without_body c) in
    ignore (run m args (Ops.body c))
  and compute u =
    let src = Ops.src u in
    match (Ops.op u, Ops.arg u) with
    | Const, Const c -> { shape = []; cells = [| { value = c; at = None } |] }
    | Param, Param { slot; _ } when List.mem_assoc slot params ->
        { (List.assoc slot params) with shape = concrete u }
    | Param, Param { bound = Some v; _ } ->
        { shape = []; cells = [| { value = (v :> Dtype.const); at = None } |] }
    | (Param | Buffer | Alloc), _ -> storage m u
    | (Reshape | Expand | Pad | Shrink | Permute), _ ->
        movement u (value (List.hd src))
    | (Detach | Contiguous_backward), _ -> value (List.hd src)
    | (Stage | Copy), _ -> fresh (value (List.hd src))
    | Reduce, Reduce { op; num_axes } ->
        reduce u op num_axes (value (List.hd src))
    | Stack, _ -> stack u (List.map value src)
    | Index, _ -> (
        match List.map value src with
        | t :: idxs -> index u t idxs
        | [] -> fail "a gather has a source")
    | Store, _ -> (
        match src with
        | dst :: v :: _ ->
            store m (value dst) (value v);
            none
        | _ -> fail "a store has a destination and a value")
    | After, _ ->
        List.iter (fun s -> ignore (value s)) (List.tl src);
        (* A join holds the value of the state under it. *)
        if Ops.is_join u then value (List.hd src)
        else reread m (value (List.hd src))
    | Sink, _ ->
        List.iter (fun s -> ignore (value s)) src;
        none
    | Call, _ ->
        call value u;
        none
    | End, _ -> (
        (* A loop calls its body once a trip, each trip reading what the
           previous ones stored into the storage the call writes; other
           storage it reads in the state its argument names. *)
        match src with
        | [ c; r ] when Ops.op c = Call ->
            let written =
              Option.value (Ops.Tbl.find_opt loops u) ~default:[]
            in
            (* The state an argument names: through movements and joins. *)
            let rec state a =
              match Ops.op a with
              | After when Ops.is_join a -> state (Ops.nth a 0)
              | After | Buffer | Param | Alloc -> a
              | _ -> ( match Ops.src a with x :: _ -> state x | [] -> a)
            in
            let rec before o s =
              s != o && Ops.op s = After && (Ops.nth s 0 == o || before o (state (Ops.nth s 0)))
            in
            let args = Ops.src_without_body c in
            (* A carry: an argument of storage the loop writes, in the latest
               state of it the call is given; an older state is read as it
               was. *)
            let carried a =
              List.memq (Ops.storage_base a) written
              && not
                   (List.exists
                      (fun b ->
                        Ops.storage_base b == Ops.storage_base a
                        && before (state a) (state b))
                      args)
            in
            for k = 0 to count r - 1 do
              call
                (fun a -> if carried a then reread m (value a) else value a)
                (Ops.substitute ~calls:Skip ~pass:Fixed_point c
                   [ (r, Ops.int k) ])
            done;
            none
        | _ -> fail "cannot evaluate an end of %d sources" (List.length src))
    | Bitcast, _
      when Dtype.itemsize (Ops.dtype u)
           <> Dtype.itemsize (Ops.dtype (List.hd src)) ->
        rebytes u (value (List.hd src))
    | op, _ when Op.Set.mem op Op.Set.alu || op = Cast || op = Bitcast ->
        elementwise u (List.map value src)
    | op, _ -> fail "cannot evaluate a %a" Op.pp op
  in
  value u

let elements t = Array.map (fun c -> c.value) t.cells

let eval ?(buffers = []) u =
  elements (run { buffers; written = Written.create 64 } [] u)

let writes ?(buffers = []) u =
  let m = { buffers; written = Written.create 64 } in
  ignore (run m [] u);
  Written.fold
    (fun (scratch, s, i) v acc ->
      match v with
      | #Dtype.value as v when not scratch -> (s, i, v) :: acc
      | _ -> acc)
    m.written []
  |> List.sort (fun (s0, i0, _) (s1, i1, _) -> compare (s0, i0) (s1, i1))
