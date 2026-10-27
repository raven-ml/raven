(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module D = Nx.Dtype
module A = Nx_array
module L = Nx_array.Layout
module Ops = Tolk.Ops
module Shape = Tolk.Shape
module Engine = Tolk_engine
module Arith = Lower_arith

exception Declined of string

let decline fmt = Printf.ksprintf (fun s -> raise (Declined s)) fmt
let numel s = Array.fold_left ( * ) 1 s
let ints l = List.map (fun n -> Ops.Int n) l
let shape_of (f : _ Nx.Prim.form) = Iarray.to_array (L.shape f.layout)

(* The device a placement holds a whole value on, if it is one. *)
let device_of p =
  let s = Nx.Placement.devices p in
  List.find_opt
    (fun r -> Nx.Placement.equal p (Nx.Placement.device s r))
    (Nx.rigs s)

(* Storage

   A parameter binds the run of its storage that its layout reads: from the
   element at or below its first position whose bits start on 16 bytes,
   through its last. Views at other offsets of one layout then share a
   program, and the run's start fixes the phase at which tolk aligns its
   accesses. *)

let alignment = 16

(* The run [(start, length)], in elements, of a layout with an element. *)
let run (type v s) (dt : (v, s) D.t) l =
  let lo, hi = L.span l in
  let per = Int.max 1 (8 * alignment / D.bits dt) in
  let start = lo - (lo mod per) in
  (start, hi - start)

let within (type v s) (dt : (v, s) D.t) l =
  if L.numel l = 0 then l
  else
    let start, _ = run dt l in
    L.v ~offset:(L.offset l - start) ~strides:(L.strides l) (L.shape l)

(* The run's bytes: its first, and their number. *)
let run_bytes (type v s) (dt : (v, s) D.t) l =
  let start, n = run dt l in
  let first = start * D.bits dt / 8 in
  (first, D.bytes dt (start + n) - first)

(* The bytes by which byte [first] of [b] lies past 16 bytes of memory. Memory
   that has no address the host computes on starts its allocation on its
   device's alignment, so the bytes before [b] in it place [b]. *)
let phase b first =
  let rec find j =
    if j >= alignment || Rig.Buffer.aligned b ~at:(first - j) alignment then j
    else find (j + 1)
  in
  match find 0 with
  | j -> j
  | exception Invalid_argument _ ->
      (Rig.Buffer.offset b + first) land (alignment - 1)

let parts (type v s) (dt : (v, s) D.t) = if D.is D.Complex dt then 2 else 1
let packed (type v s) (dt : (v, s) D.t) = D.bits dt < 8
let tdtype (type v s) (dt : (v, s) D.t) = Arith.dtype (D.Any dt)

(* Packed elements

   A sub-byte dtype's elements lie [8 / bits] to a byte, the first in the
   lowest bits. A node holds one element a byte ({!Lower_arith}), so the
   storage's bytes are unpacked where a parameter reads them and a result is
   packed where it is stored. *)

let uint8 u n = Shape.const_like ~dtype:Uint8 u (`Int (Tolk.Bigint.of_int n))

(* The shifts [0; b; ...; (k - 1) b] along the last axis of [[m; k]]. *)
let shifts b k m =
  Shape.expand
    (Shape.reshape (Shape.arange ~step:b ~dtype:Uint8 (k * b)) (ints [ 1; k ]))
    (ints [ m; k ])

(* The element a node holds for the low bits [c] of a byte. *)
let of_code (type v s) (dt : (v, s) D.t) c =
  match dt with
  | D.Int4 ->
      let int n = Shape.const_like ~dtype:Int8 c (`Int (Tolk.Bigint.of_int n)) in
      Ops.sub (Ops.bitwise_xor (Ops.bitcast c Int8) (int 8)) (int 8)
  | _ -> c

(* The low bits that store a node's element [u]. *)
let code (type v s) (dt : (v, s) D.t) u =
  match dt with
  | D.Int4 -> Ops.bitwise_and (Ops.bitcast u Uint8) (uint8 u 15)
  | _ -> Ops.bitwise_and u (uint8 u ((1 lsl D.bits dt) - 1))

let unpack dt bytes =
  let m = Shape.max_numel bytes and b = D.bits dt in
  let k = 8 / b in
  let wide = Shape.expand (Shape.reshape bytes (ints [ m; 1 ])) (ints [ m; k ]) in
  let codes =
    Ops.bitwise_and (Ops.shr wide (shifts b k m)) (uint8 wide ((1 lsl b) - 1))
  in
  of_code dt (Shape.reshape codes (ints [ m * k ]))

let pack dt u =
  let n = Shape.max_numel u and b = D.bits dt in
  let k = 8 / b in
  let m = (n + k - 1) / k in
  let flat = Shape.reshape (code dt u) (ints [ n ]) in
  let flat =
    if m * k = n then flat
    else Shape.pad flat [ Some (Ops.Int 0, Ops.Int ((m * k) - n)) ]
  in
  Shape.rop
    (Ops.shl (Shape.reshape flat (ints [ m; k ])) (shifts b k m))
    Tolk.Op.Add [ 1 ]

(* Views

   A layout over a flat node of its storage's elements, [[m]] or [[m; 2]] for
   a complex dtype: a slice where it is contiguous, and otherwise a gather at
   [offset + Σ index·stride]. *)

let zeros (type v s) (dt : (v, s) D.t) shape =
  let dims = Arith.dims (D.Any dt) shape in
  Shape.expand
    (Shape.reshape
       (Ops.const ~dtype:(tdtype dt) (`Int Tolk.Bigint.zero))
       (List.map (fun _ -> Ops.Int 1) dims))
    dims

let viewed (type v s) (dt : (v, s) D.t) flat l =
  let shape = Iarray.to_array (L.shape l) in
  let n = numel shape and off = L.offset l in
  if L.is_contiguous l then
    let keep =
      Some (Ops.Int off, Ops.Int (off + n))
      :: (if parts dt = 2 then [ None ] else [])
    in
    Shape.reshape (Shape.shrink flat keep) (Arith.dims (D.Any dt) shape)
  else
    let r = Array.length shape in
    let full = ints (Array.to_list shape) in
    let along i n =
      let v = Shape.arange ~dtype:Weak_int n in
      let step = L.stride l i in
      let v = if step = 1 then v else Ops.mul v (Ops.int step) in
      Shape.expand
        (Shape.reshape v (List.init r (fun j -> Ops.Int (if j = i then n else 1))))
        full
    in
    let base =
      Shape.const_like ~dtype:Weak_int
        (Shape.expand (Ops.int 0) full)
        (`Int (Tolk.Bigint.of_int off))
    in
    let at = List.fold_left Ops.add base (List.mapi along (Array.to_list shape)) in
    Ops.index flat [ at ]

(* [storage ~device ~phase dt l] is a buffer node for the run of a value of
   [dt] laid out by [l], an element at least, and the value's node over it. *)
let storage ~device ~phase (type v s) (dt : (v, s) D.t) l =
  let _, n = run dt l in
  if packed dt then
    let b = Ops.new_buffer ~phase device (snd (run_bytes dt l)) Uint8 in
    (b, viewed dt (unpack dt b) (within dt l))
  else
    let b = Ops.new_buffer ~phase device (n * parts dt) (tdtype dt) in
    let flat = if parts dt = 2 then Shape.reshape b (ints [ n; 2 ]) else b in
    (b, viewed dt flat (within dt l))

(* A value of every set computed on the host, then on [rig] in C order. *)
let computed rig x =
  Nx.Repr.read x (fun a ->
      let l = A.layout a in
      let a = if L.is_contiguous l && L.offset l = 0 then a else A.copy a in
      A.Any (A.to_device rig a))

(* [store dt o u] is the buffer [o] once the node [u] of a value of [dt] is
   stored into it, packed for a sub-byte [dt]. *)
let store dt o u =
  let v =
    if packed dt then pack dt u
    else Shape.reshape u (ints [ Shape.max_numel u ])
  in
  Ops.after o [ Ops.store o v ]

(* Lowering *)

type out =
  | Stored of Ops.t  (** Stored into this buffer. *)
  | Lent of int  (** Stored into argument [k]'s storage, which it takes. *)
  | Taken of int  (** Argument [k] unchanged, its storage taken. *)
  | Empty  (** No element. *)
  | Kept of value  (** A constant of every set, returned as it is. *)

(* A check: the buffers its first failure's index and each of its data there
   are stored into, one element each, and its failure. *)
type check = {
  shape : int array;
  first : Ops.t;
  data : (D.any * Ops.t) list;
  fail : int array -> A.any list -> exn;
}

type lowered = {
  sink : Ops.t option;  (** [None] where no result is stored. *)
  params : Ops.t option array;  (** Each argument's buffer, if it has one. *)
  outs : out array;
  checks : check list;  (** In the program's order. *)
  bound : (Ops.t * Rig.Buffer.t list) list;  (** The captures' storage. *)
}

type state = {
  rig : Rig.t;
  device : Ops.device;
  mutable captures : (value * Ops.t) list;
  mutable bound : (Ops.t * Rig.Buffer.t list) list;
}

let on_rig st p what =
  match device_of p with
  | Some r when Rig.equal r st.rig -> ()
  | Some r ->
      decline "%s on %s, where the program computes on %s" what (Rig.name r)
        (Rig.name st.rig)
  | None ->
      decline "%s placed at %s, over several devices" what
        (Format.asprintf "%a" Nx.Placement.pp p)

(* [bind st a] is the node of the array [a], bound once as a capture. *)
let bind st (A.Any a) =
  let dt = A.dtype a and l = A.layout a in
  if L.numel l = 0 then zeros dt (Iarray.to_array (L.shape l))
  else
    let first, length = run_bytes dt l in
    let b = A.buffer a in
    let buffer, u = storage ~device:st.device ~phase:(phase b first) dt l in
    st.bound <- (buffer, [ Rig.Buffer.view b ~first ~length ]) :: st.bound;
    u

(* The node of a constant: its arrays, or a value of every set computed on the
   host and copied to the program's device. *)
let capture (type v s d) st (x : (v, s, d) Nx.t) =
  match List.find_opt (fun (v, _) -> same v (V x)) st.captures with
  | Some (_, u) -> u
  | None ->
      let a =
        match Nx.Repr.shards x with
        | None -> computed st.rig x
        | Some arrays when Iarray.length arrays = 1 ->
            on_rig st (Option.get (Nx.placement x)) "a constant";
            A.Any (Iarray.get arrays 0)
        | Some _ -> decline "a constant over several devices"
      in
      let u = bind st a in
      st.captures <- (V x, u) :: st.captures;
      u

let construct = function
  | Scan _ -> "a scan"
  | Iterate _ -> "an iterate"
  | Lanes _ -> "Rune.lanes"
  | Lane_index _ -> "Rune.lane_index"
  | Call _ -> "a call"
  | Add _ -> "an addition"
  | Detach _ -> "Rune.detach"

(* Lending

   A result may take the storage of a donated argument it reads only at its
   own index, along every path: through maps that load it plainly at the
   map's shape, copies, reshapes of C order, bitcasts of one width,
   placements, a scatter's target and a contraction's [init]. Then each
   kernel that reads the argument computes the result or what it reads, and
   reads each element before the result's kernel writes over it. A result
   that reads no donated argument may take the storage of one that nothing
   reads. *)

module Is = Set.Make (Int)

(* The arguments a value reads, and those it reads only at its own index. *)
type reach = { reads : Is.t; own : Is.t }

let nothing = { reads = Is.empty; own = Is.empty }

(* A value read by operands each [(r, own)]: of reach [r], at its own index
   where [own]. *)
let combine operands =
  let reads =
    List.fold_left (fun acc (r, _) -> Is.union acc r.reads) Is.empty operands
  in
  let only k =
    List.for_all
      (fun (r, own) -> (not (Is.mem k r.reads)) || (own && Is.mem k r.own))
      operands
  in
  { reads; own = Is.filter only reads }

let contiguous_at shape (f : _ Nx.Prim.form) =
  L.is_contiguous f.layout
  && Iarray.to_array (L.shape f.layout) = Iarray.to_array shape

type reacher = { get : 'v 's 'd. ('v, 's, 'd) Nx.t -> reach }

(* Each operand's reach and whether [op] reads it at its own index. *)
let operand_reaches (type r) rc (op : r Nx.Prim.t) =
  let at : type v s d. (v, s, d) Nx.t -> bool -> reach * bool =
   fun x own -> (rc.get x, own)
  in
  let rest l = List.map (fun (Nx.Prim.Any x) -> at x false) l in
  match op with
  | Map { layout; loads; _ } ->
      List.map
        (fun (l : _ Nx.Prim.load) ->
          match l with
          | Plain x -> at x (contiguous_at (L.shape layout) (Nx.Prim.form x))
          | Padded { x; _ } -> at x false)
        (Iarray.to_list loads)
  | Copy x -> [ at x true ]
  | Move (Reshape _, x) -> [ at x (L.is_contiguous (Nx.Prim.form x).layout) ]
  | Bitcast (dt, x) -> [ at x (D.bits dt = D.bits (Nx.dtype x)) ]
  | Place (_, x) -> [ at x true ]
  | Scatter { idx; updates; into; _ } ->
      [ at idx false; at updates false; at into true ]
  | Contract { init = Some _; _ } ->
      (* Operands [a], [b], then [init], then scales and segments. *)
      let (Operands l) = Nx.Prim.operands op in
      List.mapi (fun i (Nx.Prim.Any x) -> at x (i = 2)) l
  | _ ->
      let (Operands l) = Nx.Prim.operands op in
      rest l

(* [lend ~donated ~computed ~reaches ~scattered ~fits] is the argument each
   result takes the storage of, or [-1], in three passes as RFC 0006 orders
   them: the results of a scatter that read an argument at their own index,
   then the other results that do, then the results that read no donated
   argument, each taking one that nothing reads. [donated.(k)] is whether
   argument [k] is donated, [computed.(j)] whether the program computes result
   [j], [reaches.(j)] its reach, [scattered.(j)] whether a scatter computes
   it, and [fits k j] whether argument [k] may lend to result [j]: of its
   dtype, size and placement, and read by no other result or check. An
   argument lends once. *)
let lend ~donated ~computed ~reaches ~scattered ~fits =
  let n = Array.length reaches in
  let arguments = Array.length donated in
  let lent = Array.make n (-1) and taken = Array.make arguments false in
  let pass pick =
    for j = 0 to n - 1 do
      if lent.(j) < 0 && computed.(j) then
        match List.find_opt (fun k -> (not taken.(k)) && fits k j) (pick j) with
        | Some k ->
            lent.(j) <- k;
            taken.(k) <- true
        | None -> ()
    done
  in
  let own j = Is.elements reaches.(j).own in
  pass (fun j -> if scattered.(j) then own j else []);
  pass own;
  pass (fun j ->
      if Is.exists (fun k -> donated.(k)) reaches.(j).reads then []
      else List.init arguments Fun.id);
  lent

(* [lower ~rig p forms ~layouts ~phases ~donated] is [p] over arguments of
   [forms], each laid out over its run as [layouts] says, its run at
   [phases], the arguments [donated] says lending their storage where a
   result may take it. *)
let lower ~rig (p : Program.t) (forms : form array) ~layouts ~phases ~donated =
  let st = { rig; device = Ops.Single (Rig.name rig); captures = []; bound = [] } in
  let env = Array.make p.slots None in
  let reach_env = Array.make p.slots nothing in
  let scattered = Array.make p.slots false in
  let params =
    Array.mapi
      (fun k (F f) ->
        let var = p.params.(k) in
        reach_env.(var) <- { reads = Is.singleton k; own = Is.singleton k };
        let shape = shape_of f in
        if numel shape = 0 then begin
          env.(var) <- Some (zeros f.dtype shape);
          None
        end
        else begin
          Option.iter (fun q -> on_rig st q "an argument") f.placement;
          let b, u =
            storage ~device:st.device ~phase:phases.(k) f.dtype layouts.(k)
          in
          env.(var) <- Some u;
          Some b
        end)
      forms
  in
  let node : type v s d. (v, s, d) Nx.t -> Ops.t =
   fun x ->
    match Nx.Prim.payload p.owner x with
    | Some (Var n) -> Option.get env.(n)
    | _ -> capture st x
  in
  let reach : type v s d. (v, s, d) Nx.t -> reach =
   fun x ->
    match Nx.Prim.payload p.owner x with
    | Some (Var n) -> reach_env.(n)
    | _ -> nothing
  in
  let checks = ref [] and check_reads = ref [] in
  let check (type d e) (ok : (bool, _, d) Nx.t) (data : (d, e) Nx.Prim.data)
      (fail : int array -> e -> exn) =
    if not (Rig.equal st.rig Rig.host) then
      decline "a check off the host, which raises at a later wait";
    let shape = Nx.shape ok in
    let n = numel shape in
    check_reads := reach ok :: !check_reads;
    if n > 0 then begin
      let flat (type v s) (dt : (v, s) D.t) u =
        Shape.reshape u (Arith.dims (D.Any dt) [| n |])
      in
      (* The first failure: the least index whose flag is false, or [n]. *)
      let ok = flat Nx.bool (node ok) in
      let index = Shape.arange ~dtype:Int64 n in
      let count = Shape.const_like index (`Int (Tolk.Bigint.of_int n)) in
      let masked =
        Ops.where (Ops.ne ok (uint8 ok 0)) count index
      in
      let first =
        Shape.reshape
          (List.hd
             (Lower_reduce.reduce Nx_kernel.(Monoid Min) [ 0 ] ~src:(D.Any D.Int64)
                (D.Any D.Int64) masked))
          (ints [ 1 ])
      in
      let rec walk : type e.
          (d, e) Nx.Prim.data -> (D.any * Ops.t) list * (A.any list -> e) =
        function
        | [] -> ([], fun _ -> ())
        | x :: rest ->
            check_reads := reach x :: !check_reads;
            let dt = Nx.dtype x in
            let at =
              Lower_index.gather (D.Any dt) ~axis:0 first (flat dt (node x))
            in
            let others, rest = walk rest in
            let read = function
              | a :: tl -> (A.get (A.expect dt a) (Iarray.of_list [ 0 ]), rest tl)
              | [] -> invalid_arg "Rune.jit: a check's data"
            in
            ((D.Any dt, at) :: others, read)
      in
      let data, read = walk data in
      checks := (shape, first, data, fun i a -> fail i (read a)) :: !checks
    end
  in
  let issue (type r) base (op : r Nx.Prim.t) =
    (match op with
    | Place (q, _) -> on_rig st q "a placement"
    | _ -> ());
    match op with
    | Check { ok; data; fail } -> check ok data fail
    | _ ->
        let ys =
          match Lower.op ~device:st.device node op with
          | ys -> ys
          | exception Invalid_argument why -> decline "%s" why
        in
        let r = combine (operand_reaches { get = reach } op) in
        let scatter = match op with Scatter _ -> true | _ -> false in
        Array.iteri
          (fun k y ->
            env.(base + k) <- Some y;
            reach_env.(base + k) <- r;
            scattered.(base + k) <- scatter)
          ys
  in
  Array.iter
    (function
      | Op { op; base; _ } -> issue base op
      | Do { c = Detach x; base } ->
          env.(base) <- Some (node x);
          reach_env.(base) <- reach x
      | Do { c; _ } -> decline "%s" (construct c))
    p.eqns;
  (* Results lent their arguments' storage. *)
  let reaches = Array.map (fun (V y) -> reach y) p.outputs in
  let checks_read = !check_reads in
  let scattered_out =
    Array.map
      (fun (V y) ->
        match Nx.Prim.payload p.owner y with
        | Some (Var v) -> scattered.(v)
        | _ -> false)
      p.outputs
  in
  let fits k j =
    let (F f) = forms.(k) in
    let (F g) = form p.outputs.(j) in
    let read_elsewhere =
      List.exists (fun r -> Is.mem k r.reads) checks_read
      || Array.exists Fun.id
           (Array.mapi (fun i r -> i <> j && Is.mem k r.reads) reaches)
    in
    donated.(k) && Option.is_some params.(k) && (not read_elsewhere)
    && Nx.Dtype.equal f.dtype g.dtype
    && numel (shape_of f) = numel (shape_of g)
    && (match (f.placement, g.placement) with
       | Some a, Some b -> Nx.Placement.equal a b
       | _ -> false)
    && L.is_contiguous layouts.(k)
    && L.offset layouts.(k) = 0
    && phases.(k) = 0
  in
  let computed =
    Array.map
      (fun (V y) ->
        match Nx.Prim.payload p.owner y with Some (Var _) -> true | _ -> false)
      p.outputs
  in
  let lent =
    lend ~donated ~computed ~reaches ~scattered:scattered_out ~fits
  in
  let stores = ref [] in
  let buffer (F f) =
    let n = numel (shape_of f) in
    if packed f.dtype then Ops.new_buffer st.device (D.bytes f.dtype n) Uint8
    else Ops.new_buffer st.device (n * parts f.dtype) (tdtype f.dtype)
  in
  let stored (F f as form) u =
    if numel (shape_of f) = 0 then Empty
    else
      let o = buffer form in
      stores := store f.dtype o u :: !stores;
      Stored o
  in
  let outs =
    Array.mapi
      (fun j (V y) ->
        let f = Nx.Prim.form y in
        match Nx.Prim.payload p.owner y with
        | Some (Var v) -> (
            match f.placement with
            | None -> decline "a result of every set"
            | Some q -> (
                on_rig st q "a result";
                match lent.(j) with
                | k when k >= 0 && v = p.params.(k) -> Taken k
                | k when k >= 0 ->
                    let b = Option.get params.(k) in
                    stores := store f.dtype b (node y) :: !stores;
                    Lent k
                | _ -> stored (F f) (node y)))
        | _ when Option.is_none f.placement -> Kept (V y)
        | _ -> stored (F f) (capture st y))
      p.outputs
  in
  let one (type v s) (dt : (v, s) D.t) u =
    let f : (v, s, _) Nx.Prim.form =
      { dtype = dt; layout = L.contiguous (Iarray.of_list [ 1 ]); placement = None }
    in
    let o = buffer (F f) in
    stores := store dt o u :: !stores;
    o
  in
  let checks =
    List.rev_map
      (fun (shape, first, data, fail) ->
        let first = one D.Int64 first in
        let data =
          List.map (fun (D.Any dt as d, u) -> (d, one dt u)) data
        in
        { shape; first; data; fail })
      !checks
  in
  let sink =
    match !stores with [] -> None | s -> Some (Ops.sink (List.rev s))
  in
  { sink; params; outs; checks; bound = st.bound }

(* Linking *)

let debug = Tolk.Setting.bool ~reach:Process "RUNE_JIT_DEBUG" false

let report fmt =
  if Tolk.Setting.value debug then Printf.eprintf ("rune.jit: " ^^ fmt ^^ "\n%!")
  else Printf.ifprintf stderr fmt

(* A kernel searched at width [width], each candidate timed on [rig], on the
   slots of the first one timed. *)
let search rig width k =
  report "searched a kernel at width %d" width;
  let slots = ref None in
  let link call = Engine.link_call ~devices:(fun _ -> rig) call in
  let time ~vars s =
    let slots =
      match !slots with
      | Some b -> b
      | None ->
          let b = Engine.slots s in
          slots := Some b;
          b
    in
    Engine.time ~vars s slots
  in
  Tolk.Search.beam_search ~link ~time ~clock:Engine.clock width k

(* The settings a caller may change around a call that change what tolk
   compiles: each setting that shapes it, and whether the engine links each
   kernel as a submission of its own, as it does for its reports. *)
type settings = (string * string) list * bool

let settings () =
  (Tolk.Setting.shaping (), Tolk.Setting.value Tolk.Setting.debug >= 2)

(* A program linked for one layout and phase of each argument and one value of
   the settings: the linked schedule, its variables, and the slot of each
   argument and output, [-1] for none. *)
type linked = {
  schedule : (Engine.t * (string * int) list) option;
  inputs : int;
  leaf_slots : int array;
  out_slots : int array;
  outs : out array;
  checks : (check * int * int list) list;
      (** Each check, the slot of its index and of each datum. *)
}

type variant = {
  layouts : L.t array;  (** Each argument's over its run ({!within}). *)
  phases : int array;
  donated : bool array;
  settings : settings;
  linked : (linked, string) result;
      (** [Error why] where the lowering or tolk declines the variant, which
          then evaluates. *)
}

let link rig l =
  match l.sink with
  | None ->
      {
        schedule = None;
        inputs = 0;
        leaf_slots = Array.map (fun _ -> -1) l.params;
        out_slots = Array.map (fun _ -> -1) l.outs;
        outs = l.outs;
        checks = [];
      }
  | Some sink ->
      let present = Ops.Tbl.create 16 in
      List.iter
        (fun u ->
          if Ops.op u = Tolk.Op.Buffer then Ops.Tbl.replace present u ())
        (Ops.toposort ~calls:Enter sink);
      let order = ref [] and next = ref 0 in
      let slot u =
        order := u :: !order;
        incr next;
        !next - 1
      in
      let leaf_slots =
        Array.map
          (function
            | Some b when Ops.Tbl.mem present b -> slot b | Some _ | None -> -1)
          l.params
      in
      let out_slots =
        Array.map
          (function Stored o -> slot o | Lent _ | Taken _ | Empty | Kept _ -> -1)
          l.outs
      in
      let checks =
        List.map
          (fun c -> (c, slot c.first, List.map (fun (_, o) -> slot o) c.data))
          l.checks
      in
      let linear, vars =
        Tolk.Schedule.create_linear_with_vars ~capturing:true sink
      in
      let linear =
        Tolk.Jit.jit_lower ~search:(search rig)
          ~targets:(fun _ -> Engine.target rig)
          ~held_bufs:(List.map fst l.bound) ~inputs:(List.rev !order) linear
      in
      let schedule = Engine.link ~devices:(fun _ -> rig) ~bound:l.bound linear in
      {
        schedule = Some (schedule, vars);
        inputs = !next;
        leaf_slots;
        out_slots;
        outs = l.outs;
        checks;
      }

(* Programs *)

type t = {
  name : string;
  rig : Rig.t;
  home : form;  (** A placed form on [rig], at whose set arguments of every
                    set are placed. *)
  program : Program.t;
  forms : form array;
  lock : Mutex.t;
  variants : variant list Atomic.t;
}

(* The program's device, and a form placed whole on it: its arguments', or
   else its results'. *)
let device (p : Program.t) forms =
  let first = ref None in
  let consider (F f as form) =
    match (!first, f.placement) with
    | None, Some q -> (
        match device_of q with
        | Some r -> first := Some (r, form)
        | None ->
            decline "a value placed at %s, over several devices"
              (Format.asprintf "%a" Nx.Placement.pp q))
    | _ -> ()
  in
  Array.iter consider forms;
  Array.iter (fun v -> consider (form v)) p.outputs;
  match !first with
  | Some (r, home) when Rig.computes r -> (r, home)
  | Some (r, _) -> decline "a value on %s, which computes nothing" (Rig.name r)
  | None -> decline "no value placed"

(* What a program the compiler takes must not raise to compile: an operation it
   does not lower, or tolk refusing the graph, as a target whose compiler is
   missing does. *)
let declined = function
  | Declined why -> Some why
  | Invalid_argument why | Failure why -> Some why
  | _ -> None

let same_variant layouts phases donated settings v =
  v.phases = phases && v.donated = donated && v.settings = settings
  && Array.for_all2 L.equal v.layouts layouts

let variant t layouts phases donated settings =
  let find () =
    List.find_opt
      (same_variant layouts phases donated settings)
      (Atomic.get t.variants)
  in
  match find () with
  | Some v -> v
  | None -> (
      Mutex.protect t.lock @@ fun () ->
      match find () with
      | Some v -> v
      | None ->
          (* A declined variant is kept; another failure, as an interrupted
             search, keeps nothing. *)
          let linked =
            match
              link t.rig
                (lower ~rig:t.rig t.program t.forms ~layouts ~phases ~donated)
            with
            | l -> Ok l
            | exception e -> (
                match declined e with
                | Some why ->
                    report "evaluates a variant uncompiled: %s" why;
                    Error why
                | None -> raise e)
          in
          let v = { layouts; phases; donated; settings; linked } in
          Atomic.set t.variants (v :: Atomic.get t.variants);
          v)

let make ~name p forms =
  match
    let rig, home = device p forms in
    let layouts =
      Array.map (fun (F f) -> within f.dtype (L.contiguous (L.shape f.layout))) forms
    in
    let phases = Array.map (fun _ -> 0) forms in
    let donated = Array.map (fun _ -> false) forms in
    ignore (lower ~rig p forms ~layouts ~phases ~donated);
    {
      name;
      rig;
      home;
      program = p;
      forms;
      lock = Mutex.create ();
      variants = Atomic.make [];
    }
  with
  | t -> Ok t
  | exception e -> (
      match declined e with Some why -> Error why | None -> raise e)

(* Calls *)

(* [consumed ~by args f] is [f arrays take] under a consumer of the placed
   values of [args], [arrays k] the arrays of placed argument [k] and
   [take k a] the memory of its array [a] for a result, if it may take it
   ({!Nx.Prim.Consumer.take}). *)
let consumed ~by args f =
  let placed =
    List.filter
      (fun k -> match args.(k) with V x -> Option.is_some (Nx.placement x))
      (List.init (Array.length args) Fun.id)
  in
  match placed with
  | [] ->
      let none _ = invalid_arg (by ^ ": no argument is placed") in
      f none (fun k _ -> none k)
  | k0 :: _ ->
      let (V x0) = args.(k0) in
      let f0 = Nx.Prim.form x0 in
      let any k =
        let (V y) = args.(k) in
        match Nx.Prim.coerce { f0 with dtype = Nx.dtype y } y with
        | Some y -> Nx.Prim.Any y
        | None -> invalid_arg (by ^ ": arguments on several sets")
      in
      let position = Array.make (Array.length args) (-1) in
      List.iteri (fun i k -> position.(k) <- i) placed;
      Nx.Prim.Consumer.with_ ~by (List.map any placed) (fun c ->
          f
            (fun k -> Nx.Prim.Consumer.arrays c position.(k))
            (fun k (A.Any a) ->
              Option.map A.buffer
                (Nx.Prim.Consumer.take c position.(k) (A.dtype a) (A.layout a))))

(* [x], of every set, computed whole on [rig] at [home]'s set, where an
   operation that reads it computes it. *)
let placed rig (F home) (V x as v) =
  match (Nx.placement x, home.placement) with
  | None, Some q -> (
      let f = { home with dtype = Nx.dtype x; layout = (Nx.Prim.form x).layout } in
      match Nx.Prim.coerce f x with
      | Some x ->
          V (Nx.place (Nx.Placement.device (Nx.Placement.devices q) rig) x)
      | None -> v)
  | _ -> v

let result rig (F f) buffer =
  let shape = shape_of f in
  let buffer =
    match buffer with
    | Some b -> b
    | None -> Rig.Buffer.create rig (D.bytes f.dtype (numel shape))
  in
  let q = Option.get f.placement in
  V
    (Nx.Repr.unsafe_of_array (Nx.Placement.devices q)
       (A.v f.dtype (L.contiguous (Iarray.of_array shape)) buffer))

(* Evaluation *)

(* [x] as a value over [arrays], its arrays under a consumer. *)
let over (V x) arrays =
  let f = Nx.Prim.form x in
  match f.placement with
  | None -> V x
  | Some q ->
      V
        (Nx.Repr.of_shards q
           (Iarray.of_array (Array.map (A.expect f.dtype) arrays)))

(* [p] evaluated over the arrays of a consumer of [args]: the donors it
   claimed are dead, so the arrays of every placed argument stand for it, a
   donor beside its handle included. Each result is copied, since no value
   over the arrays may outlive the consumer. *)
let evaluated p args arrays =
  let values =
    Array.mapi
      (fun k (V x as v) ->
        if Option.is_some (Nx.placement x) then over v (arrays k) else v)
      args
  in
  Array.map (fun (V y) -> V (Nx.copy y)) (eval p values)

let evaluate ~name p args =
  if Array.exists (fun (V x) -> Nx.Prim.donated x) args then
    consumed ~by:name args (fun arrays _ -> evaluated p args arrays)
  else
    (* A result that is an argument is a copy of it. *)
    Array.map
      (fun (V y as r) -> if Array.exists (same r) args then V (Nx.copy y) else r)
      (eval p args)

(* The element of the one-element array over [b], read on the host. *)
let element (type v s) (dt : (v, s) D.t) b : A.any =
  let a = A.v dt (L.contiguous (Iarray.of_list [ 1 ])) b in
  A.Any (A.to_device Rig.host a)

(* The index of C-order position [i] of [shape]. *)
let unravel shape i =
  let r = Array.length shape in
  let index = Array.make r 0 and rest = ref i in
  for d = r - 1 downto 0 do
    index.(d) <- !rest mod shape.(d);
    rest := !rest / shape.(d)
  done;
  index

(* Raises [c]'s failure if it failed: its index is below its element count. *)
let answer slots ((c : check), first, data) =
  let at = A.expect D.Int64 (element D.Int64 (List.hd slots.(first))) in
  let i = Int64.to_int (A.get at (Iarray.of_list [ 0 ])) in
  if i < numel c.shape then
    let data =
      List.map2
        (fun (D.Any dt, _) s -> element dt (List.hd slots.(s)))
        c.data data
    in
    raise (c.fail (unravel c.shape i) data)

(* A buffer for one element of [dt]. *)
let one rig (type v s) (dt : (v, s) D.t) = Rig.Buffer.create rig (D.bytes dt 1)

let rec run t args =
  let p = t.program in
  let args = Array.map (placed t.rig t.home) args in
  let donated = Array.map (fun (V x) -> Nx.Prim.donated x) args in
  consumed ~by:t.name args @@ fun arrays take ->
  let leaves = Array.mapi (fun k _ -> (arrays k).(0)) args in
  let layouts =
    Array.map (fun (A.Any a) -> within (A.dtype a) (A.layout a)) leaves
  in
  let phases =
    Array.map
      (fun (A.Any a) ->
        let l = A.layout a in
        if L.numel l = 0 then 0
        else phase (A.buffer a) (fst (run_bytes (A.dtype a) l)))
      leaves
  in
  match (variant t layouts phases donated (settings ())).linked with
  | Error _ -> evaluated p args arrays
  | Ok v -> linked t v args leaves take

(* [linked t v args leaves take] runs the variant [v] of [t] on [args], whose
   arrays under the call's consumer are [leaves]. *)
and linked t v args leaves take =
  let p = t.program in
  let slots = Array.make v.inputs [] in
  let run_of k =
    let (A.Any a) = leaves.(k) in
    let first, length = run_bytes (A.dtype a) (A.layout a) in
    Rig.Buffer.view (A.buffer a) ~first ~length
  in
  (* A lending argument's memory, or a copy of it where nx's consumer does not
     give its memory, as where another value reads it. *)
  let lending = Array.make (Array.length args) None in
  let lend k =
    match lending.(k) with
    | Some b -> b
    | None ->
        let b =
          match take k leaves.(k) with
          | Some b -> b
          | None ->
              let src = run_of k in
              let dst = Rig.Buffer.create t.rig (Rig.Buffer.length src) in
              Rig.Buffer.copy ~src ~dst;
              report "an argument copied: its memory is shared";
              dst
        in
        lending.(k) <- Some b;
        b
  in
  Array.iter (function Lent k | Taken k -> ignore (lend k) | _ -> ()) v.outs;
  Array.iteri
    (fun k s ->
      if s >= 0 then
        slots.(s) <-
          [ (match lending.(k) with Some b -> b | None -> run_of k) ])
    v.leaf_slots;
  let outputs = Array.map form p.outputs in
  let fresh (F f) =
    Rig.Buffer.create t.rig (D.bytes f.dtype (numel (shape_of f)))
  in
  let buffers =
    Array.mapi
      (fun j s ->
        match v.outs.(j) with
        | Lent k | Taken k -> lending.(k)
        | Stored _ ->
            let b = fresh outputs.(j) in
            slots.(s) <- [ b ];
            Some b
        | Empty | Kept _ -> None)
      v.out_slots
  in
  List.iter
    (fun ((c : check), first, data) ->
      slots.(first) <- [ one t.rig D.Int64 ];
      List.iter2
        (fun (D.Any dt, _) s -> slots.(s) <- [ one t.rig dt ])
        c.data data)
    v.checks;
  Option.iter (fun (l, vars) -> Engine.run ~vars l slots) v.schedule;
  List.iter (answer slots) v.checks;
  (* A result of every set is a constant: the program declines one it
     computes. *)
  Array.mapi
    (fun j (F f as o) ->
      match f.placement with
      | None -> p.outputs.(j)
      | Some _ -> result t.rig o buffers.(j))
    outputs

