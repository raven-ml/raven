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

(* What stores wrote, by place: scratch or not, slot and index. A map, so
   that a state keeps the memory it was left in while later stores run. *)
module Written = Map.Make (struct
  type t = bool * int * int

  let compare (s0, l0, i0) (s1, l1, i1) =
    match Bool.compare s0 s1 with
    | 0 -> ( match Int.compare l0 l1 with 0 -> Int.compare i0 i1 | c -> c)
    | c -> c
end)

(* [clock] counts the stores run, the time of a state. *)
type memory = {
  buffers : (int * Dtype.value array) list;
  mutable written : Dtype.const Written.t;
  mutable clock : int;
}

let initial m at : Dtype.const =
  match List.assoc_opt at.slot m.buffers with
  | _ when at.scratch -> `Invalid
  | None -> `Invalid
  | Some a when at.index < Array.length a -> (a.(at.index) :> Dtype.const)
  | Some _ -> fail "element %d is outside memory %d" at.index at.slot

let key at = (at.scratch, at.slot, at.index)

let current m written at =
  match Written.find_opt (key at) written with
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
      | Some at, _ ->
          m.written <- Written.add (key at) value.cells.(k).value m.written
      | None, `Invalid -> ()
      | None, _ -> fail "a store's destination is not storage")
    dst.cells;
  m.clock <- m.clock + 1

(* [reread m written t] is [t] as memory [written] holds it. *)
let reread m written t =
  let cell c =
    match c.at with
    | Some at -> { c with value = current m written at }
    | None -> c
  in
  { t with cells = Array.map cell t.cells }

(* The value of an effect, which nothing reads. *)
let none = { shape = []; cells = [||] }

(* [run m params u] is the value of [u], each parameter of slot [k] standing for
   the value [params] gives [k], if any. *)
let rec run m params u =
  let values = ref (Ops.Tbl.create 64) in
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
  (* The states the Afters listing an effect are over, with their storage and
     the effects listed before it there. *)
  let over = Ops.Tbl.create 8 in
  let rec listed d =
    if Ops.op d = Sink then List.concat_map listed (Ops.src d) else [ d ]
  in
  List.iter
    (fun a ->
      if Ops.op a = After && not (Ops.is_join a) then
        ignore
          (List.fold_left
             (fun earlier e ->
               Ops.Tbl.replace over e
                 ((Ops.storage_base a, List.hd (Ops.src a), earlier)
                 :: Option.value (Ops.Tbl.find_opt over e) ~default:[]);
               e :: earlier)
             []
             (List.concat_map listed (List.tl (Ops.src a)))))
    (Ops.toposort ~calls:Skip u);
  let same a b =
    a == b
    ||
    match (Ops.arg a, Ops.arg b) with
    | Param { slot = s; _ }, Param { slot = t; _ } ->
        Ops.op a = Param && Ops.op b = Param && s = t
    | _ -> false
  in
  let writes_into e s =
    let call_writes c =
      match Shape.written_params (Ops.body c) with
      | None -> true
      | Some w ->
          List.exists
            (fun i ->
              same (Ops.storage_base (List.nth (Ops.src_without_body c) i)) s)
            w
    in
    match Ops.op e with
    | Store -> same (Ops.storage_base (List.hd (Ops.src e))) s
    | Call -> call_writes e
    | End when Ops.op (List.hd (Ops.src e)) = Call ->
        call_writes (List.hd (Ops.src e))
    | _ -> true
  in
  (* The state an argument names: through movements and joins. *)
  let rec state a =
    match Ops.op a with
    | After when Ops.is_join a -> state (Ops.nth a 0)
    | After | Buffer | Param | Alloc -> a
    | _ -> ( match Ops.src a with x :: _ -> state x | [] -> a)
  in
  (* Whether the state [p] has the effect [e] in its chain: a listing of [e]
     again on a later state, which orders nothing. *)
  let rec in_chain p e =
    let x = state p in
    Ops.op x = After
    && (List.memq e (List.concat_map listed (List.tl (Ops.src x)))
       || in_chain (Ops.nth x 0) e)
  in
  (* The time and memory each effect left, and each state that a write left
     or holds. Storage before any store has none. *)
  let left = Ops.Tbl.create 64 in
  let is_effect e = List.mem (Ops.op e) Op.[ Store; Call; End ] in
  let active = Ops.Tbl.create 64 in
  let rec value u =
    match Ops.Tbl.find_opt !values u with
    | Some v -> v
    | None ->
        if Ops.Tbl.mem active u then
          fail "an effect is listed over a state that needs it";
        Ops.Tbl.replace active u ();
        (* An effect runs after the state each After listing it is over, and
           the effects listed before its first place there, where it writes
           that After's storage, unless that state holds it. *)
        List.iter
          (fun (s, p, earlier) ->
            if
              writes_into u s
              && (not (in_chain p u))
              && not (List.memq u earlier)
            then (
              ignore (value p);
              List.iter
                (fun d -> if writes_into d s then ignore (value d))
                (List.rev earlier)))
          (List.rev (Option.value (Ops.Tbl.find_opt over u) ~default:[]));
        let v = compute u in
        if is_effect u then Ops.Tbl.replace left u (m.clock, m.written);
        Ops.Tbl.remove active u;
        Ops.Tbl.replace !values u v;
        v
  (* [call read c] runs the body of the call [c] on its arguments, as [read]
     gives them. *)
  and call read c =
    (* An argument the call writes is the storage as the call runs, once
       every argument has run. *)
    let written =
      Option.value (Shape.written_params (Ops.body c)) ~default:[]
    in
    let args = List.map read (Ops.src_without_body c) in
    let arg k a = (k, if List.mem k written then reread m m.written a else a) in
    ignore (run m (List.mapi arg args) (Ops.body c))
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
    | After, _ -> (
        (* A state lands after the state it is over, and holds the latest of
           that state, those its writes of its storage leave and those of its
           storage its effects read. A join holds
           the value of the state under it, read once its other sources have
           run. *)
        let under = value (List.hd src) in
        List.iter (fun s -> ignore (value s)) (List.tl src);
        let s = Ops.storage_base u in
        let latest best d =
          match Ops.Tbl.find_opt left d with
          | Some (t, _) as l when is_effect d && writes_into d s -> (
              match best with Some (b, _) when b >= t -> best | _ -> l)
          | _ -> best
        in
        (* A state of its storage that a listed effect reads leaves the
           storage no older. *)
        let read best d =
          Ops.Nodes.fold
            (fun x best ->
              match Ops.Tbl.find_opt left x with
              | Some (t, _) as l
                when Ops.op x = After
                     && (not (Ops.is_join x))
                     && same (Ops.storage_base x) s -> (
                  match best with Some (b, _) when b >= t -> best | _ -> l)
              | _ -> best)
            (Ops.backward_slice ~calls:Skip d)
            best
        in
        let effects = List.concat_map listed (List.tl src) in
        let named =
          List.fold_left read
            (List.fold_left latest
               (Ops.Tbl.find_opt left (state (List.hd src)))
               effects)
            effects
        in
        match named with
        | _ when Ops.is_join u -> under
        | None -> under
        | Some ((_, written) as l) ->
            Ops.Tbl.replace left u l;
            reread m written under)
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
            let args = Ops.src_without_body c in
            (* The storage the loop writes: its call's written arguments', or,
               where its body does not show them, that of the Afters listing
               it. *)
            let written =
              match Shape.written_params (Ops.body c) with
              | Some w -> List.map (fun i -> Ops.storage_base (List.nth args i)) w
              | None -> Option.value (Ops.Tbl.find_opt loops u) ~default:[]
            in
            let rec before o s =
              s != o && Ops.op s = After
              && (state (Ops.nth s 0) == o || before o (state (Ops.nth s 0)))
            in
            (* A state of the storage the loop writes, not older than one
               another argument names, is read as the trips before left it,
               and so is each value computed from it. *)
            let over =
              List.filter_map
                (fun (s, p, _) -> if Ops.op u = End then Some (s, p) else None)
                (Option.value (Ops.Tbl.find_opt over u) ~default:[])
            in
            let live n =
              List.exists (same (Ops.storage_base n)) written
              && not
                   (List.exists
                      (fun b ->
                        Ops.storage_base b == Ops.storage_base n
                        && before (state n) (state b))
                      (args @ List.filter_map
                                (fun (s, p) ->
                                  if s == Ops.storage_base n then Some p else None)
                                over))
            in
            (* The states the arguments read, and the effects under them,
               run once, before the first trip. *)
            let rec states n =
              match Ops.op n with
              | After when Ops.is_join n -> List.iter states (Ops.src n)
              | After | Buffer | Param | Alloc -> ignore (value n)
              | _ -> List.iter states (Ops.src n)
            in
            List.iter states args;
            let saved = Ops.Tbl.copy !values in
            for k = 0 to count r - 1 do
              let c =
                Ops.substitute ~calls:Skip ~pass:Fixed_point c
                  [ (r, Ops.int k) ]
              in
              let trip = Ops.Tbl.copy saved in
              let rec retrip n =
                match Ops.op n with
                | (After | Buffer | Param | Alloc) when live n ->
                    let v =
                      match Ops.Tbl.find_opt saved n with
                      | Some v -> v
                      | None -> value n
                    in
                    Ops.Tbl.replace trip n (reread m m.written v)
                | After | Buffer | Param | Alloc | Const -> ()
                | _ ->
                    Ops.Tbl.remove trip n;
                    List.iter retrip (Ops.src n)
              in
              List.iter retrip (Ops.src_without_body c);
              values := trip;
              call value c
            done;
            values := saved;
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
  elements (run { buffers; written = Written.empty; clock = 0 } [] u)

let writes ?(buffers = []) u =
  let m = { buffers; written = Written.empty; clock = 0 } in
  ignore (run m [] u);
  Written.fold
    (fun (scratch, s, i) v acc ->
      match v with
      | #Dtype.value as v when not scratch -> (s, i, v) :: acc
      | _ -> acc)
    m.written []
  |> List.sort (fun (s0, i0, _) (s1, i1, _) -> compare (s0, i0) (s1, i1))
