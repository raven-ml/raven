open Tolk

(* An element carries the memory it views, if any, so that a store through a
   movement of storage knows where it writes, and on which device. Call-local
   storage numbers its slots apart from parameters and buffers. *)
type place = { scratch : bool; slot : int; index : int; device : int }
type cell = { value : Dtype.const; at : place option }
type tensor = { shape : int list; cells : cell array }

let fail fmt = Format.kasprintf invalid_arg fmt

let concrete u =
  List.map
    (function
      | Ops.Int n -> n
      | Sym _ -> fail "cannot evaluate a %a of symbolic shape" Op.pp (Ops.op u))
    (Ops.shape u)

let size shape = List.fold_left ( * ) 1 shape

(* Device ranges

   A value on several devices may depend on a device range: on device [k], the
   range stands for [k]. *)

let device_ranges u =
  Ops.Nodes.fold
    (fun r acc -> if Ops.axis_type r = Device then r :: acc else acc)
    (Ops.ranges u) []

let count r = Bigint.to_int (Ops.to_z (Ops.nth r 0))

let on_device k u =
  Ops.ssimplify
    (Ops.substitute u (List.map (fun r -> (r, Ops.int k)) (device_ranges u)))

let int ~device = function
  | Ops.Int n -> n
  | Sym u -> (
      match on_device device u with
      | Int n -> n
      | Sym _ -> fail "a movement is symbolic")

(* The device ranges an argument of a movement holds. *)
let marg_ranges u =
  let sints : Ops.movement -> Ops.sint list = function
    | Reshape s | Expand s -> s
    | Pad p | Shrink p -> List.concat_map (fun (a, b) -> [ a; b ]) p
    | Permute _ | Flip _ -> []
  in
  List.concat_map
    (function Ops.Int _ -> [] | Sym s -> device_ranges s)
    (sints (Ops.marg u))

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
  let lead = List.length shape - List.length t.shape in
  gather shape t (fun idx ->
      Some
        (List.map2
           (fun n i -> if n = 1 then 0 else i)
           t.shape
           (List.filteri (fun a _ -> a >= lead) idx)))

let movement ~device u t =
  let int = int ~device in
  let shape = concrete u in
  let zero = Dtype.const (Ops.dtype u) (`Int Bigint.zero) in
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

(* [unshard u axes shards] is the whole value of [u], whose device [k] holds
   [shards]'s [k]th part: along each axis of [axes], the part at the position
   that axis's range takes on [k]. *)
let unshard u axes shards =
  let rngs = List.tl (Ops.src u) in
  let n =
    match List.concat_map device_ranges rngs with
    | r :: _ -> count r
    | [] -> fail "cannot evaluate a value sharded within a kernel"
  in
  let shards = match shards with [ t ] -> List.init n (fun _ -> t) | s -> s in
  let parts =
    List.mapi
      (fun k t -> (List.map (fun r -> int ~device:k (Sym r)) rngs, t))
      shards
  in
  let holds idx (positions, t) =
    List.for_all2
      (fun axis p -> List.nth idx axis / List.nth t.shape axis = p)
      axes positions
  in
  let shape = concrete u in
  let cell k =
    let idx = coords shape k in
    match List.find_opt (holds idx) parts with
    | Some (_, t) ->
        let local =
          List.mapi
            (fun axis i ->
              if List.mem axis axes then i mod List.nth t.shape axis else i)
            idx
        in
        t.cells.(offset t.shape local)
    | None -> fail "an element of a sharded value is on no device"
  in
  { shape; cells = Array.init (size shape) cell }

(* Memory *)

type memory = {
  buffers : (int * Dtype.value array) list;
  written : (bool * int * int, Dtype.const) Hashtbl.t;
}

let initial m at : Dtype.const =
  match List.assoc_opt at.slot m.buffers with
  | _ when at.scratch -> `Invalid
  | None -> `Invalid
  | Some a when at.index < Array.length a -> (a.(at.index) :> Dtype.const)
  | Some _ -> fail "element %d is outside memory %d" at.index at.slot

let key at = (at.scratch, at.slot, at.index)

let current m at =
  match Hashtbl.find_opt m.written (key at) with
  | Some v -> v
  | None -> initial m at

let storage m u =
  match Ops.arg u with
  | Param { slot; size; device; addrspace; _ } when addrspace <> Some Alu ->
      let n = Option.value size ~default:1 in
      let shape = concrete u in
      List.init (devices device) (fun k ->
          let cell i =
            let scratch = Ops.op u = Alloc in
            let at = { scratch; slot; index = (k * n) + i; device = k } in
            { value = initial m at; at = Some at }
          in
          { shape; cells = Array.init n cell })
  | _ -> fail "cannot evaluate a variable"

(* Each element of the destination takes the value of the device whose memory it
   views, a value on one device standing for every device. *)
let store m dst value =
  List.iter
    (fun d ->
      let value = List.map (broadcast d.shape) value in
      let source at =
        match value with [ v ] -> v | value -> List.nth value at.device
      in
      Array.iteri
        (fun k c ->
          match (c.at, c.value) with
          | Some at, _ ->
              Hashtbl.replace m.written (key at) (source at).cells.(k).value
          | None, `Invalid -> ()
          | None, _ -> fail "a store's destination is not storage")
        d.cells)
    dst

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
    | Range, Range { axis_type = Device; _ } ->
        List.init (count u) (fun k ->
            {
              shape = [];
              cells = [| { value = `Int (Bigint.of_int k); at = None } |];
            })
    | Param, Param { slot; _ } when List.mem_assoc slot params ->
        let shape = concrete u in
        List.map (fun t -> { t with shape }) (List.assoc slot params)
    | (Param | Buffer | Alloc), _ -> storage m u
    | (Reshape | Expand | Pad | Shrink | Permute | Flip), _ ->
        let v = value (List.hd src) in
        let v =
          match (marg_ranges u, v) with
          | r :: _, [ t ] -> List.init (count r) (fun _ -> t)
          | _ -> v
        in
        List.mapi (fun device t -> movement ~device u t) v
    | (Detach | Contiguous_backward), _ -> value (List.hd src)
    | Stage, _ -> List.map fresh (value (List.hd src))
    | Reduce, Reduce { op; num_axes } ->
        List.map (reduce u op num_axes) (value (List.hd src))
    | Stack, _ -> across (stack u) (List.map value src)
    | Index, _ ->
        across
          (function
            | t :: idxs -> index u t idxs | [] -> fail "a gather has a source")
          (List.map value src)
    | Mselect, Shard i -> [ List.nth (value (List.hd src)) i ]
    | Mstack, _ -> List.concat_map value src
    | Copy, Device d ->
        let t = List.hd (value (List.hd src)) in
        List.init (devices (Some d)) (fun _ -> fresh t)
    | Unshard, Axes axes -> [ unshard u axes (value (List.hd src)) ]
    | Allreduce, Allreduce { op; device } ->
        let shards = value (List.hd src) in
        let t = List.hd shards in
        (* Devices are never none, so the fold starts from the first, and an
           operation without an identity, such as [Or], folds too. *)
        let dt = Ops.dtype u in
        let cell k =
          let values = List.map (fun s -> s.cells.(k).value) shards in
          let combine acc v =
            held dt (Ops.exec_alu ~truncate_output:false op dt [ acc; v ])
          in
          {
            value = List.fold_left combine (List.hd values) (List.tl values);
            at = None;
          }
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
        (* The body's parameter of a sharded argument holds its parts. *)
        let part a = if Ops.op a = Unshard then Ops.nth a 0 else a in
        let args = List.mapi (fun k a -> (k, value (part a))) (List.tl src) in
        ignore (run m args (List.hd src));
        []
    | Bitcast, _
      when Dtype.itemsize (Ops.dtype u)
           <> Dtype.itemsize (Ops.dtype (List.hd src)) ->
        List.map (rebytes u) (value (List.hd src))
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
    (fun (scratch, s, i) v acc ->
      match v with
      | #Dtype.value as v when not scratch -> (s, i, v) :: acc
      | _ -> acc)
    m.written []
  |> List.sort (fun (s0, i0, _) (s1, i1, _) -> compare (s0, i0) (s1, i1))
