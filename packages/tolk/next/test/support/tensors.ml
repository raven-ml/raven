open Tolk_next

(* An element carries the memory it views, if any, so that a store through a
   movement of storage knows where it writes. *)
type cell = { value : Dtype.const; at : (int * int) option }
type tensor = { shape : int list; cells : cell array }

let fail fmt = Format.kasprintf invalid_arg fmt

let concrete u =
  List.map
    (function
      | Ops.Int n -> n
      | Sym _ -> fail "cannot evaluate a %a of symbolic shape" Op.pp (Ops.op u))
    (Ops.shape u)

let int = function Ops.Int n -> n | Sym _ -> fail "a movement is symbolic"
let size shape = List.fold_left ( * ) 1 shape

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

let broadcast shape t =
  let lead = List.length shape - List.length t.shape in
  gather shape t (fun idx ->
      Some
        (List.map2
           (fun n i -> if n = 1 then 0 else i)
           t.shape
           (List.filteri (fun a _ -> a >= lead) idx)))

let movement u t =
  let shape = concrete u in
  let zero = Dtype.const (Ops.dtype u) (`Int Z.zero) in
  match Ops.marg u with
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
  | Flip flips ->
      gather shape t (fun idx ->
          Some
            (List.map2
               (fun (n, f) i -> if f then n - 1 - i else i)
               (List.combine t.shape flips)
               idx))

(* Elements *)

let held dt : Dtype.const -> Dtype.const = function
  | #Dtype.value as v -> (
      match Dtype.const dt v with
      | #Dtype.value as v -> (Dtype.truncate dt v :> Dtype.const)
      | `Invalid -> `Invalid)
  | `Invalid -> `Invalid

let element u (values : Dtype.const list) : Dtype.const =
  let dt = Ops.dtype u in
  match (Ops.op u, values) with
  | Where, `Invalid :: _ -> `Invalid
  | Where, [ `Bool c; a; b ] -> if c then a else b
  | _, values when List.mem `Invalid values -> `Invalid
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

let fresh t =
  { t with cells = Array.map (fun c -> { c with at = None }) t.cells }

(* Devices *)

let devices = function
  | Some (Ops.Multi l) -> List.length l
  | Some (Single _) | None -> 1

(* [across f vs] applies [f] device by device, a value on one device standing
   for every device. *)
let across f vs =
  let n = List.fold_left (fun n v -> max n (List.length v)) 1 vs in
  let on k v =
    match v with
    | [ t ] -> t
    | v when List.length v = n -> List.nth v k
    | _ -> fail "values on %d and %d devices meet" (List.length v) n
  in
  List.init n (fun k -> f (List.map (on k) vs))

(* Memory *)

type memory = {
  buffers : (int * Dtype.value array) list;
  written : (int * int, Dtype.const) Hashtbl.t;
}

let initial m (slot, i) : Dtype.const =
  match List.assoc_opt slot m.buffers with
  | None -> `Invalid
  | Some a when i < Array.length a -> (a.(i) :> Dtype.const)
  | Some _ -> fail "element %d is outside memory %d" i slot

let current m at =
  match Hashtbl.find_opt m.written at with Some v -> v | None -> initial m at

let storage m u =
  match Ops.arg u with
  | Param { slot; size = Some n; device; _ } ->
      let shape = concrete u in
      List.init (devices device) (fun k ->
          let cell i =
            let at = (slot, (k * n) + i) in
            { value = initial m at; at = Some at }
          in
          { shape; cells = Array.init n cell })
  | _ -> fail "cannot evaluate a scalar parameter"

let store m dst value =
  List.iter2
    (fun d v ->
      let v = broadcast d.shape v in
      Array.iteri
        (fun k c ->
          match c.at with
          | Some at -> Hashtbl.replace m.written at v.cells.(k).value
          | None -> fail "a store's destination is not storage")
        d.cells)
    dst
    (if List.length value = 1 then List.map (fun _ -> List.hd value) dst
     else value)

let reread m v =
  List.map
    (fun t ->
      {
        t with
        cells =
          Array.map
            (fun c ->
              match c.at with
              | Some at -> { c with value = current m at }
              | None -> c)
            t.cells;
      })
    v

(* [run m params u] is the value of [u], each parameter of slot [k] standing for
   the value [params] gives [k], if any. *)
let rec run m params u =
  let values = Ops.Tbl.create 64 in
  let rec value u =
    match Ops.Tbl.find_opt values u with
    | Some v -> v
    | None ->
        let v = compute u in
        Ops.Tbl.replace values u v;
        v
  and compute u =
    let src = Ops.src u in
    match (Ops.op u, Ops.arg u) with
    | Const, Const c ->
        [ { shape = []; cells = [| { value = c; at = None } |] } ]
    | Param, Param { slot; _ } when List.mem_assoc slot params ->
        let shape = concrete u in
        List.map (fun t -> { t with shape }) (List.assoc slot params)
    | (Param | Buffer | Alloc), _ -> storage m u
    | (Reshape | Expand | Pad | Shrink | Permute | Flip), _ ->
        List.map (movement u) (value (List.hd src))
    | (Detach | Contiguous_backward), _ -> value (List.hd src)
    | Stage, _ -> List.map fresh (value (List.hd src))
    | Reduce, Reduce { op; num_axes } ->
        List.map (reduce u op num_axes) (value (List.hd src))
    | Stack, _ -> across (stack u) (List.map value src)
    | Mselect, Shard i -> [ List.nth (value (List.hd src)) i ]
    | Mstack, _ -> List.concat_map value src
    | Copy, Device d -> (
        match value (List.hd src) with
        | [ t ] -> List.init (devices (Some d)) (fun _ -> fresh t)
        | _ -> fail "a copy's source is on several devices")
    | Allreduce, Allreduce { op; device } ->
        let shards = value (List.hd src) in
        let t = List.hd shards in
        let cell k =
          let values = List.map (fun s -> s.cells.(k).value) shards in
          { value = fold op (Ops.dtype u) values; at = None }
        in
        let reduced = { t with cells = Array.init (size t.shape) cell } in
        List.init (devices (Some device)) (fun _ -> reduced)
    | Store, _ -> (
        match src with
        | dst :: v :: _ ->
            store m (value dst) (value v);
            []
        | _ -> fail "a store has a destination and a value")
    | After, _ ->
        List.iter (fun s -> ignore (value s)) (List.tl src);
        reread m (value (List.hd src))
    | Sink, _ ->
        List.iter (fun s -> ignore (value s)) src;
        []
    | Call, _ ->
        let args = List.mapi (fun k a -> (k, value a)) (List.tl src) in
        ignore (run m args (List.hd src));
        []
    | op, _ when Op.Set.mem op Op.Set.alu || op = Cast || op = Bitcast ->
        across (elementwise u) (List.map value src)
    | op, _ -> fail "cannot evaluate a %a" Op.pp op
  in
  value u

let elements v = List.map (fun t -> Array.map (fun c -> c.value) t.cells) v

let eval ?(buffers = []) u =
  elements (run { buffers; written = Hashtbl.create 64 } [] u)

let writes ?(buffers = []) u =
  let m = { buffers; written = Hashtbl.create 64 } in
  ignore (run m [] u);
  Hashtbl.fold
    (fun (s, i) v acc ->
      match v with #Dtype.value as v -> (s, i, v) :: acc | `Invalid -> acc)
    m.written []
  |> List.sort (fun (s0, i0, _) (s1, i1, _) -> compare (s0, i0) (s1, i1))
