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
   element at or below its first position whose bits start on 16 bytes, through
   its last. Views at other offsets of one layout then share a program, and the
   run's start fixes the phase at which tolk aligns its accesses. *)

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
    if start = 0 then l
    else L.v ~offset:(L.offset l - start) ~strides:(L.strides l) (L.shape l)

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

   A sub-byte dtype's elements lie [8 / bits] to a byte, the first in the lowest
   bits. A node holds one element a byte ({!Lower_arith}), so the storage's
   bytes are unpacked where a parameter reads them and a result is packed where
   it is stored. *)

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
      let int n =
        Shape.const_like ~dtype:Int8 c (`Int (Tolk.Bigint.of_int n))
      in
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
  let wide =
    Shape.expand (Shape.reshape bytes (ints [ m; 1 ])) (ints [ m; k ])
  in
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

   A layout over a flat node of its storage's elements, [[m]] or [[m; 2]] for a
   complex dtype: a slice where it is contiguous, and otherwise a gather at
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
        (Shape.reshape v
           (List.init r (fun j -> Ops.Int (if j = i then n else 1))))
        full
    in
    let base =
      Shape.const_like ~dtype:Weak_int
        (Shape.expand (Ops.int 0) full)
        (`Int (Tolk.Bigint.of_int off))
    in
    let at =
      List.fold_left Ops.add base (List.mapi along (Array.to_list shape))
    in
    Ops.index flat [ at ]

(* [storage ~device ~phase dt l] is a buffer node for the run of a value of [dt]
   laid out by [l], an element at least, and the value's node over it. *)
let storage ~device ~phase (type v s) (dt : (v, s) D.t) l =
  let _, n = run dt l in
  if packed dt then
    let b = Ops.new_buffer ~phase device (snd (run_bytes dt l)) Uint8 in
    (b, viewed dt (unpack dt b) (within dt l))
  else
    let b = Ops.new_buffer ~phase device (n * parts dt) (tdtype dt) in
    let flat = if parts dt = 2 then Shape.reshape b (ints [ n; 2 ]) else b in
    (b, viewed dt flat (within dt l))

(* [store dt o u] is the buffer [o] once the node [u] of a value of [dt] is
   stored into it, packed for a sub-byte [dt]. *)
let store dt o u =
  let v =
    if packed dt then pack dt u
    else Shape.reshape u (ints [ Shape.max_numel u ])
  in
  Ops.after o [ Ops.store o v ]

(* [read_back dt shape s] is the node of the value of [dt] and [shape] that the
   storage [s] holds in C order, as {!store} stored it. *)
let read_back dt shape s =
  let n = numel shape in
  let l = L.contiguous (Iarray.of_array shape) in
  if packed dt then viewed dt (unpack dt s) l
  else
    let flat = if parts dt = 2 then Shape.reshape s (ints [ n; 2 ]) else s in
    viewed dt flat l

(* Lowering *)

type out =
  | Stored of Ops.t  (** Stored into this buffer. *)
  | Lent of int  (** Stored into argument [k]'s storage, which it takes. *)
  | Taken of int  (** Argument [k] unchanged, its storage taken. *)
  | Copied of Rig.Buffer.t
      (** The bytes of a constant of another device, copied there into fresh
          memory on each call. *)
  | Empty  (** No element. *)

(* A check: the buffers its first failure's index and each of its data there are
   stored into, one element each, and its failure. *)
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
  uploads : (value * A.any) list ref;
      (** The constants of other devices, each with its copy on this one,
          which every lowering of the program shares. *)
  views : unit Ops.Tbl.t;
      (** The nodes of the arguments and captures, without their movements:
          views of storage. *)
  mutable loops : int;  (** The loops' steps being lowered. *)
}

let view_of st u = Ops.Tbl.replace st.views (Ops.base u) ()

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
    view_of st u;
    u

(* The array [a] of another device copied to [st]'s, once, as JAX embeds a
   constant where its program runs: its run, into memory of its own, viewed
   as [a] views its own. *)
let uploaded st (A.Any a) =
  let dt = A.dtype a and l = A.layout a in
  if L.numel l = 0 then A.Any a
  else
    let first, length = run_bytes dt l in
    let src = Rig.Buffer.view (A.buffer a) ~first ~length in
    let dst = Rig.Buffer.create st.rig length in
    Rig.Buffer.copy ~src ~dst;
    A.Any (A.v dt (within dt l) dst)

(* The node of a constant: its one array, bound once. A constant on another
   device, which the program places where it computes, is uploaded there
   once per program, and its copy bound. *)
let capture (type v s d) st (x : (v, s, d) Nx.t) =
  match List.find_opt (fun (v, _) -> same v (V x)) st.captures with
  | Some (_, u) -> u
  | None ->
      let arrays = Nx.Repr.shards x in
      if Iarray.length arrays <> 1 then
        decline "a constant over several devices";
      let a = A.Any (Iarray.get arrays 0) in
      let u =
        match device_of (Nx.placement x) with
        | Some r when Rig.equal r st.rig -> bind st a
        | Some _ ->
            let made = List.find_opt (fun (v, _) -> same v (V x)) in
            let copy =
              match made !(st.uploads) with
              | Some (_, c) -> c
              | None ->
                  let c = uploaded st a in
                  st.uploads := (V x, c) :: !(st.uploads);
                  c
            in
            bind st copy
        | None ->
            decline "a constant placed at %s, over several devices"
              (Format.asprintf "%a" Nx.Placement.pp (Nx.placement x))
      in
      st.captures <- (V x, u) :: st.captures;
      u

(* The bytes of [x], a constant result of another device than the program's,
   where a copy of them makes the result there with no kernel: one array,
   empty or contiguous in C order from a byte boundary. *)
let elsewhere (type v s d) st (x : (v, s, d) Nx.t) =
  let arrays = Nx.Repr.shards x in
  match device_of (Nx.placement x) with
  | Some r when (not (Rig.equal r st.rig)) && Iarray.length arrays = 1 ->
      let a = Iarray.get arrays 0 in
      let dt = A.dtype a and l = A.layout a in
      let bit = L.offset l * D.bits dt in
      if L.numel l = 0 then
        Some (Rig.Buffer.view (A.buffer a) ~first:0 ~length:0)
      else if L.is_contiguous l && bit mod 8 = 0 then
        let length = D.bytes dt (L.numel l) in
        Some (Rig.Buffer.view (A.buffer a) ~first:(bit / 8) ~length)
      else None
  | _ -> None

let construct_name = function
  | Scan _ -> "a scan"
  | Iterate _ -> "an iterate"
  | Lanes _ -> "Rune.lanes"
  | Lane_index _ -> "Rune.lane_index"
  | Call _ -> "a call"
  | Remat _ -> "Rune.remat"
  | Operator _ -> "Rune.root's operator"
  | Root { name; _ } -> name
  | Solve { name; _ } -> name ^ "'s linear solve"
  | Switch { name; _ } -> name
  | Add _ -> "an addition"
  | Emit _ -> "an emission"
  | Detach _ -> "Rune.detach"
  | Callback _ -> "Rune.callback"
  | Custom { name; _ } | Linear { name; _ } -> name

(* Lending

   A result may take the storage of a donated argument it reads only at its own
   index, along every path: through maps that load it plainly at the map's
   shape, copies, reshapes of C order, bitcasts of one width, placements, a
   scatter's target and a contraction's [init]. Then the result's kernel reads
   each element before it writes over it; another kernel that reads the
   argument runs before ({!assign}). A result that reads no donated argument
   may take the storage of one that nothing reads. *)

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
   result takes the storage of, or [-1], in three passes: the results of a
   scatter that read an argument at their own index, then the other results
   that do, then the results that read no donated argument, each taking one
   that nothing reads. [donated.(k)] is whether argument [k] is donated,
   [computed.(j)] whether the program computes result [j],
   [reaches.(j)] its reach, [scattered.(j)] whether a scatter computes it, and
   [fits k j] whether argument [k] may lend to result [j]: of its dtype, size
   and placement, and read by no check. An argument lends once. *)
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
  let unread k = Array.for_all (fun r -> not (Is.mem k r.reads)) reaches in
  pass (fun j ->
      if Is.exists (fun k -> donated.(k)) reaches.(j).reads then []
      else List.filter unread (List.init arguments Fun.id));
  lent

(* Assignments

   A lent result is an assignment to its argument's storage, as tinygrad's
   assign is: every other value of the program that reads it, a result or a
   check, reads that storage once it is stored. A kernel that reads an
   argument as it was runs before the store over it, and one that reads a
   stored result runs after that store, so the stores need an order in which
   both hold. A momentum step has one: the velocity's store reads the
   parameters as they were, and the parameters' store reads the velocity where
   it was stored. Two results that each read the other's argument as it was
   have none: the latest result of such a cycle gives up the argument it
   took, and the arguments are lent again without that pair, until an order
   exists. *)

(* The first cycle of the graph [succ] over [0 .. n - 1], if one exists. *)
let cycle succ =
  let n = Array.length succ in
  let state = Array.make n 0 and path = ref [] and found = ref None in
  let rec visit i =
    if !found = None then
      match state.(i) with
      | 1 ->
          let rec upto acc = function
            | j :: rest -> if j = i then i :: acc else upto (j :: acc) rest
            | [] -> acc
          in
          found := Some (upto [] !path)
      | 2 -> ()
      | _ ->
          state.(i) <- 1;
          path := i :: !path;
          List.iter visit succ.(i);
          path := List.tl !path;
          state.(i) <- 2
  in
  for i = 0 to n - 1 do
    if state.(i) = 0 then visit i
  done;
  !found

type assigned = {
  stored : Ops.t option array;  (** Each lent result's storage, stored. *)
  read : int -> Ops.t -> Ops.t;
      (** [read j u] is the node [u] of result [j], or of a check where [j] is
          [-1], reading each lent result but [j] from its storage. *)
}

(* [assign ~nodes ~checks ~store ~view ~buffer lent] is [Ok] the assignments of
   the results [lent] lends to, or [Error j] where their stores have no order,
   [j] the latest lent result of a cycle: [nodes.(j)] is result [j]'s node,
   [None] where no kernel computes it, as for an argument kept; [checks] holds
   each check's nodes, which the stores order as a result's; [store j k u]
   stores [u], result [j]'s value, over argument [k]'s storage, [view j s]
   reads result [j] from the storage once [s] stored it, and [buffer k] is
   argument [k]'s buffer. *)
let assign ~nodes ~checks ~store ~view ~buffer lent =
  let n = Array.length nodes in
  let assigned j = lent.(j) >= 0 && Option.is_some nodes.(j) in
  let node j = Option.get nodes.(j) in
  let stored = Array.make n None in
  (* [u] reading each lent result it reaches but [self] from its storage; of
     two lent results of one node, the later reads the earlier's. *)
  let rec read self u =
    let slice = lazy (Ops.backward_slice ~calls:Skip u) in
    let reached i =
      let w = node i in
      if w == u then i < self || not (self >= 0 && assigned self)
      else Ops.Nodes.mem w (Lazy.force slice)
    in
    let pairs =
      List.filter_map
        (fun i ->
          if i <> self && assigned i && reached i then
            Some (node i, view i (store_of i))
          else None)
        (List.init n Fun.id)
    in
    Ops.substitute ~calls:Skip ~pass:Once u pairs
  and store_of j =
    match stored.(j) with
    | Some s -> s
    | None ->
        let s = store j lent.(j) (read j (node j)) in
        stored.(j) <- Some s;
        s
  in
  (* [succ.(i)] holds the results whose kernels run after [i]'s: after its
     store where [i] is lent, before the store over an argument it reads as
     it was. *)
  let olds = Ops.Tbl.create 8 and stores = Ops.Tbl.create 8 in
  for j = 0 to n - 1 do
    if assigned j then begin
      Ops.Tbl.replace olds (buffer lent.(j)) j;
      Ops.Tbl.replace stores (store_of j) j
    end
  done;
  let checks = Array.of_list checks in
  let succ = Array.make (n + Array.length checks) [] in
  (* The edges of vertex [r], whose nodes [us] read as result [self]. *)
  let edges r ~self us =
    let after x =
      match Ops.Tbl.find_opt stores x with
      | Some i when i <> r ->
          succ.(i) <- r :: succ.(i);
          false
      | _ -> true
    in
    List.iter
      (fun u ->
        List.iter
          (fun x ->
            match Ops.Tbl.find_opt olds x with
            | Some i when i <> r -> succ.(r) <- i :: succ.(r)
            | _ -> ())
          (Ops.toposort ~calls:Skip ~gate:after (read self u)))
      us
  in
  for r = 0 to n - 1 do
    if Option.is_some nodes.(r) then edges r ~self:r [ node r ]
  done;
  Array.iteri (fun c us -> edges (n + c) ~self:(-1) us) checks;
  match cycle succ with
  | None -> Ok { stored; read }
  | Some c ->
      let lent_in = List.filter (fun j -> j < n && assigned j) c in
      Error (List.fold_left max (-1) lent_in)

(* Programs

   A program lowers equation by equation into an environment of its own: each
   variable's node, its reach, and whether a scatter computes it. A construct
   whose programs compute where it is called, a call, a remat, a custom rule's
   primal and a linear call's function, lowers its program into an environment
   whose parameters are its operands' nodes.

   A scan or an iterate lowers to one loop ([repeat]) over its trips, calling
   its step once per trip. The parts of the step that read no trip's parameter
   are computed once, before the loop. A construct's totals start at its seeds,
   or at the identity where it has none ([starts]), and are carried from trip to
   trip. Their values after the last trip are its sums, among its results. An
   iterate's loop continues while its stop is false and no check has failed, a
   flag the engine reads before each trip. A loop still running after [max]
   trips fails a check of the program. Each check a trip makes, a nested loop's
   included, is carried across trips, so the loop keeps each check's first
   failure and its data, which become the program's checks. *)

type env = {
  p : Program.t;
  nodes : Ops.t option array;
  reaches : reach array;
  scattered : bool array;
}

let env_of (p : Program.t) =
  {
    p;
    nodes = Array.make p.slots None;
    reaches = Array.make p.slots nothing;
    scattered = Array.make p.slots false;
  }

let var env x =
  match Nx.Prim.payload env.p.owner x with Some (Var n) -> n | _ -> -1

let node st env x =
  match var env x with
  | n when n >= 0 -> Option.get env.nodes.(n)
  | _ -> capture st x

let reach env x =
  match var env x with n when n >= 0 -> env.reaches.(n) | _ -> nothing

let scattered_at env x =
  match var env x with n when n >= 0 -> env.scattered.(n) | _ -> false

(* The checks a program makes: the index of each one's first failure and its
   data there, and what each reads, in the program's order, newest first. *)
type made = {
  shape : int array;
  first : Ops.t;
  data : (D.any * Ops.t) list;
  fail : int array -> A.any list -> exn;
}

(* The checks a program makes, in order, newest first, what they read, and
   whether one has failed so far: [1] where one has, [0] otherwise, as an
   [int64] of one element. An iterate runs on only while none has, as the
   steps written out stop at the first failure. *)
type checks = {
  mutable made : made list;
  mutable read : reach list;
  mutable failed : Ops.t;
}

let int64 n =
  let c = Ops.const ~dtype:Int64 (`Int (Tolk.Bigint.of_int n)) in
  Shape.reshape c (ints [ 1 ])

let int32 n =
  let c = Ops.const ~dtype:Int32 (`Int (Tolk.Bigint.of_int n)) in
  Shape.reshape c (ints [ 1 ])

let fresh_checks failed = { made = []; read = []; failed }

(* [m] made, after the checks [cs] holds. *)
let note cs (m : made) =
  let count = int64 (numel m.shape) in
  let failing = Ops.where (Ops.lt m.first count) (int64 1) (int64 0) in
  cs.failed <- Ops.maximum cs.failed failing;
  cs.made <- m :: cs.made

(* The first failure of [ok], a boolean node of [n] elements: the least index
   whose flag is false, or [n]. *)
let first_failure n ok =
  let index = Shape.arange ~dtype:Int64 n in
  let count = Shape.const_like index (`Int (Tolk.Bigint.of_int n)) in
  let masked = Ops.where (Ops.ne ok (uint8 ok 0)) count index in
  Shape.reshape
    (List.hd
       (Lower_reduce.reduce
          Nx_kernel.(Monoid Min)
          [ 0 ] ~src:(D.Any D.Int64) (D.Any D.Int64) masked))
    (ints [ 1 ])

(* The check that [ok], a boolean node of one element, holds, raising
   [failure] otherwise. *)
let raises cs ok failure =
  let first = first_failure 1 (Shape.reshape ok (ints [ 1 ])) in
  note cs { shape = [| 1 |]; first; data = []; fail = (fun _ _ -> failure ()) }

let check st cs env (type d e) (ok : (bool, _, d) Nx.t)
    (data : (d, e) Nx.Prim.data) (fail : int array -> e -> exn) =
  if not (Rig.equal st.rig Rig.host) then
    decline "a check off the host, which raises at a later wait";
  let shape = Nx.shape ok in
  let n = numel shape in
  cs.read <- reach env ok :: cs.read;
  if n > 0 then begin
    let flat (type v s) (dt : (v, s) D.t) u =
      Shape.reshape u (Arith.dims (D.Any dt) [| n |])
    in
    let first = first_failure n (flat Nx.bool (node st env ok)) in
    let rec walk : type e.
        (d, e) Nx.Prim.data -> (D.any * Ops.t) list * (A.any list -> e) =
      function
      | [] -> ([], fun _ -> ())
      | x :: rest ->
          cs.read <- reach env x :: cs.read;
          let dt = Nx.dtype x in
          let at =
            Lower_index.gather (D.Any dt) ~axis:0 first (flat dt (node st env x))
          in
          let others, rest = walk rest in
          let read = function
            | a :: tl -> (A.get (A.expect dt a) (Iarray.of_list [ 0 ]), rest tl)
            | [] -> invalid_arg "Rune.jit: a check's data"
          in
          ((D.Any dt, at) :: others, read)
    in
    let data, read = walk data in
    note cs { shape; first; data; fail = (fun i a -> fail i (read a)) }
  end

let issue st cs env (type r) base (op : r Nx.Prim.t) =
  (match op with Place (q, _) -> on_rig st q "a placement" | _ -> ());
  match op with
  | Check { ok; data; fail } -> check st cs env ok data fail
  | _ ->
      let ys =
        match Lower.op ~device:st.device (fun x -> node st env x) op with
        | ys -> ys
        | exception Invalid_argument why -> decline "%s" why
      in
      let r = combine (operand_reaches { get = (fun x -> reach env x) } op) in
      let scatter = match op with Scatter _ -> true | _ -> false in
      Array.iteri
        (fun k y ->
          env.nodes.(base + k) <- Some y;
          env.reaches.(base + k) <- r;
          env.scattered.(base + k) <- scatter)
        ys

(* The node of a value of [f]'s form filled with [-0.], the identity that keeps
   every sign: a tangent map's additions that no differentiation made. *)
let identity st form =
  let (V x) = Loops.filled ~zero:false form in
  capture st x

(* Loops

   A scan lowers as one loop of the program: a range of as many trips as it has
   rows, around one call of its step, lowered once as the call's body. Its
   parameters stand for one trip's carry and rows; its captures are the
   enclosing program's nodes, so each part of the body that reads none of the
   trip's parameters is computed once, before the loop ({!Loop.cut}). On trip
   [i] the body reads row [i] of each stacked input and writes row [i] of each
   stacked output, row [n - 1 - i] where the scan is reversed, rows lying 16 bytes of memory apart, as the body's vector
   accesses require. Each carry, and each sum of the step's additions, is one
   buffer that each trip updates in place, once every kernel that reads it
   ran, as the schedule orders them: a next carry that reads the carry
   elsewhere than at its own index is computed into storage of its own first,
   and so is, each trip, the latest of carries whose next values read each
   other in a cycle, such as two carries that swap.

   An iterate lowers as a loop of at most [max] trips: its stop at the initial
   carry is computed before the loop, and each trip computes the step, then the
   stop at the carry it stores, and stores the flag that the stop does not hold
   yet, which the engine reads before the next trip. Once the trips are done, a
   stop that still fails raises the iterate's failure, a check of the
   program.

   A step's checks are carried values of the loop: the first trip in which one
   fails keeps each check's first failure and its data there, and later trips
   leave them, so the loop's checks raise the first failure of the first step
   that fails. The lowering keeps whether a check of the program has failed so
   far, which every iterate's stop reads, and a loop carries it where its trip
   reads it or checks, so an iterate stops at the program's first failure. *)

(* How a node reads a value: not at all, at each element's own index only
   (through elementwise operations, casts and bitcasts of one width, reshapes
   and contiguous markers), or otherwise. *)
type how = Apart | Own | Other

let keeps_index u =
  let sized v =
    not (List.exists (Tolk.Dtype.equal (Ops.dtype v)) Tolk.Dtype.weaks)
  in
  match Ops.op u with
  | Tolk.Op.Cast | Tolk.Op.Bitcast ->
      let v = Ops.nth u 0 in
      sized u && sized v && Ops.element_size u = Ops.element_size v
  | Tolk.Op.Reshape | Tolk.Op.Stage -> true
  | o -> Tolk.Op.Set.mem o Tolk.Op.Set.elementwise

(* [reads ~from u] is how [u] reads [from]. The partial application walks each
   node once. *)
let reads ~from =
  let memo = Ops.Tbl.create 64 in
  let rec go u =
    if u == from then Own
    else
      match Ops.Tbl.find_opt memo u with
      | Some r -> r
      | None ->
          let srcs = List.map go (Ops.src u) in
          let r =
            if List.for_all (( = ) Apart) srcs then Apart
            else if keeps_index u && not (List.mem Other srcs) then Own
            else Other
          in
          Ops.Tbl.add memo u r;
          r
  in
  go

(* The elements of [u]'s dtype that rows of [m] of them take, apart enough that
   each starts on 16 bytes of memory. *)
let stride u m =
  let per = Int.max 1 (alignment / Ops.element_size u) in
  (m + per - 1) / per * per

let window b start m =
  Shape.shrink b [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + int m)) ]

(* [a] where [c], a boolean of one element, holds, and [b] elsewhere. *)
let pick c a b =
  let rank = List.length (Shape.shape a) in
  let c = Shape.reshape c (List.init rank (fun _ -> Ops.Int 1)) in
  Ops.where (Shape.expand c (Shape.shape a)) a b

let blank u =
  Shape.expand
    (Ops.const ~dtype:(Ops.dtype u) (`Int Tolk.Bigint.zero))
    (Shape.shape u)

(* A loop's trip count, an [int32] node of one element, clamped to [0] to
   [n]. *)
let counted n t =
  let t = Shape.reshape t (ints [ 1 ]) in
  Ops.minimum
    (Ops.maximum t (Shape.const_like t (`Int Tolk.Bigint.zero)))
    (Shape.const_like t (`Int (Tolk.Bigint.of_int n)))

(* Of [u], [n] rows of a value of [d] along axis 0, the first [t] rows, [t] a
   count from [counted], then zeros. Where [reversed] holds, row [j] below [t]
   is [u]'s row [t - 1 - j], so that a loop running rows forward runs a
   reversed scan's from its last row that runs. *)
let leading d ~reversed ~n t u =
  let parts = Arith.parts d u in
  let shape = Shape.shape (List.hd parts) in
  let along v =
    let lead = Ops.Int n :: List.map (fun _ -> Ops.Int 1) (List.tl shape) in
    Shape.expand (Shape.reshape v lead) shape
  in
  let j = Shape.arange ~dtype:Int64 n in
  let t = Shape.expand (Ops.cast t Int64) (ints [ n ]) in
  if reversed then
    let last = Ops.sub t (Shape.const_like t (`Int Tolk.Bigint.one)) in
    Lower_index.gather d ~axis:0 (along (Ops.sub last j)) u
  else
    let kept = along (Ops.lt j t) in
    Arith.of_parts d (List.map (fun p -> Ops.where kept p (blank p)) parts)

(* A loop's checks, carried: for each check its trip made, in order, its first
   failure in the first trip that failed, or its element count, and its data
   there. Each trip after the first that failed leaves them, so the loop raises
   the first failure of the first trip that failed, as the steps written out
   would. A check of a loop inside the step is one of the step's. *)
let check_inits (made : made list) =
  List.concat_map
    (fun (m : made) ->
      int64 (numel m.shape) :: List.map (fun (_, u) -> blank u) m.data)
    made

(* The carried checks' next values, given their values [carried], whether a
   check failed before the trip ([before]) and by its end ([after]), and the
   checks [made] of this trip. *)
let check_nexts ~before ~after carried (made : made list) =
  let at = ref 0 in
  let next u =
    let c = carried.(!at) in
    incr at;
    pick before c (u c)
  in
  List.concat_map
    (fun (m : made) ->
      let n = int64 (numel m.shape) in
      let first = next (fun _ -> pick after m.first n) in
      first :: List.map (fun (_, u) -> next (fun c -> pick after u c)) m.data)
    made

(* The loop's checks, each a check of the step with its carried first failure
   and data, of the carried values' finals [finals]. *)
let check_finals cs (made : made list) finals =
  let at = ref 0 in
  let take () =
    let u = finals.(!at) in
    incr at;
    u
  in
  List.iter
    (fun (m : made) ->
      let first = take () in
      let data = List.map (fun (d, _) -> (d, take ())) m.data in
      note cs { m with first; data })
    made

let empty u = Shape.max_numel u = 0

(* Where an iterate's lanes stop apart, their flags the carry [j] of [carried]:
   the index of the first still false, an [int64] node of one element, or the
   flags' count, and the lanes it names once read back. *)
let running forms carried = function
  | None -> None
  | Some j -> (
      let (F f) = forms.(j) in
      let shape = shape_of f in
      let n = numel shape in
      match D.equal_witness f.dtype Nx.bool with
      | Some Type.Equal when n > 0 ->
          let flags = Shape.reshape carried.(j) (ints [ n ]) in
          let named = function
            | [ a ] ->
                let i = A.get (A.expect D.Int64 a) (Iarray.of_list [ 0 ]) in
                Program.lane_names shape (Int64.to_int i)
            | _ -> invalid_arg "Rune.jit: an iterate's lanes"
          in
          Some (first_failure n flags, named)
      | _ -> None)

(* Whether an equation or an output of [p] names its variable [n]. *)
let names (p : Program.t) n =
  let is (type v s d) (x : (v, s, d) Nx.t) =
    match Nx.Prim.payload p.owner x with Some (Var m) -> m = n | _ -> false
  in
  let named (V x) = is x in
  Array.exists named p.outputs
  || Array.exists
       (function
         | Op { op; _ } ->
             let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
             List.exists (fun (Nx.Prim.Any x) -> is x) xs
         | Do { c; _ } -> List.exists named (operands c))
       p.eqns

(* A value's node, reach, and whether a scatter computes it. *)
let operand st env (V x) = (node st env x, reach env x, scattered_at env x)

let on_device st u =
  if Option.is_none (Ops.device u) then Ops.copy_to_device u st.device else u

(* A differentiated remat's operands through a barrier, as JAX lowers one
   through an optimization barrier: each operand the program computes is stored
   into storage of its own, which the remat reads once every such operand is
   stored. Its values over those operands are then none the program has
   elsewhere, so the compiler merges none of them with the forward pass's, and
   they wait for the cotangents among the operands, so they compute in the
   backward pass. An operand that is a view of storage, as an argument or a
   capture moved or not, is read in place, uncopied: what the remat computes
   from such operands alone is the forward pass's. *)
let barrier st operands =
  let computed (u, _, _) =
    (not (empty u)) && not (Ops.Tbl.mem st.views (Ops.base u))
  in
  let stored =
    Array.map
      (fun ((u, _, _) as o) ->
        if not (computed o) then None
        else
          let b = Ops.new_buffer st.device (Shape.max_numel u) (Ops.dtype u) in
          let view = Shape.reshape b (Shape.shape u) in
          Some (Ops.after b [ Ops.store view (on_device st u) ]))
      operands
  in
  Array.mapi
    (fun i ((u, r, _) as o) ->
      match stored.(i) with
      | None -> o
      | Some s ->
          let others =
            List.filteri (fun j _ -> j <> i) (Array.to_list stored)
          in
          let s = Ops.after s (List.filter_map Fun.id others) in
          (Shape.reshape s (Shape.shape u), r, false))
    operands

let rec program st cs env =
  Array.iter
    (function
      | Op { op; base; _ } -> issue st cs env base op
      | Do { c; base } -> construct st cs env c base)
    env.p.eqns

and construct st cs env c base =
  let bind outs =
    Array.iteri
      (fun k (u, r, sc) ->
        env.nodes.(base + k) <- Some u;
        env.reaches.(base + k) <- r;
        env.scattered.(base + k) <- sc)
      outs
  in
  match c with
  | Detach x -> bind [| operand st env (V x) |]
  | Remat { body; args; seeds; differentiated = true } when st.loops = 0 ->
      (* Inside a loop's step, the remat is a call body's, which shares no node
         with the forward loop. *)
      let operands = barrier st (Array.map (operand st env) args) in
      bind (threaded_over st cs env body operands seeds)
  | Call { body; args; seeds; _ } | Remat { body; args; seeds; _ } ->
      bind (threaded st cs env body args seeds)
  | Linear { apply = Some a; args; captures; _ } ->
      bind (inline st cs env a (Array.append args captures))
  | Custom { primal; derivative; args; captures; seeds; _ } ->
      (* The default's results: the primal's, its residuals computed, then a
         tangent map's additions, which no differentiation made. *)
      let out = threaded st cs env primal (Array.append args captures) seeds in
      let tadds =
        match derivative with
        | Tangent t ->
            let n = Array.length t.outputs in
            Array.init (Array.length t.adds) (fun k ->
                let v = t.outputs.(n - Array.length t.adds + k) in
                (identity st (form v), nothing, false))
        | Pullback _ -> [||]
      in
      bind (Array.append out tadds)
  | Root { solution; _ } -> bind (Array.map (operand st env) solution)
  | Solve { system; transposed; b; data; captures; _ } ->
      (* The linear solve's program, its operator applications calls of [J]
         or [Jᵀ], its additions dropped: the default's. *)
      let body ~transposed:t ~lanes =
        system.operator ~transposed:(t <> transposed) ~lanes
      in
      let p = Program.fill system.solve ~id:system.id body in
      let out =
        threaded st cs env p (Array.concat [ b; data; captures ]) [||]
      in
      bind (Array.sub out 0 (Array.length b))
  | Scan _ | Iterate _ -> bind (loop st cs env c)
  | Switch s -> bind (switch st cs env c s)
  | c -> decline "%s" (construct_name c)

(* A switch: one loop per branch, of at most one trip, under a flag that holds
   where the index names the branch and no check of the program has failed.
   Each loop carries the results and sums the one before left, so the taken
   branch's reach the end. A branch reads the captures as its call's
   parameters, so an untaken one computes nothing. An index
   outside the branches is a check, made before any branch's. *)
and switch st cs env c (s : Program.switch) =
  let forms = Program.forms c in
  let nb = Array.length s.branches in
  let _, sum_forms, feeds = Program.sums (Array.to_list s.branches) in
  let ns = Array.length sum_forms in
  let nr = Array.length forms - ns in
  let (V i) = s.index in
  let index = Shape.reshape (node st env i) (ints [ 1 ]) in
  let at k = Shape.const_like index (`Int (Tolk.Bigint.of_int k)) in
  let no = Ops.ne (int64 0) (int64 0) in
  let inside = Ops.where (Ops.lt index (at 0)) no (Ops.lt index (at nb)) in
  let read a =
    Int32.to_int (A.get (A.expect D.Int32 a) (Iarray.of_list [ 0 ]))
  in
  let fail _ = function
    | [ a ] ->
        Invalid_argument
          (Printf.sprintf "%s: index %d outside 0 .. %d" s.name (read a) (nb - 1))
    | _ -> invalid_arg "Rune.jit: a switch's index"
  in
  cs.read <- reach env i :: cs.read;
  note cs
    {
      shape = [| 1 |];
      first = first_failure 1 inside;
      data = [ (D.Any D.Int32, index) ];
      fail;
    };
  let reach = loop_reach env c in
  let caps = Array.map (fun (V x) -> node st env x) s.captures in
  let results =
    Array.init nr (fun j ->
        let (F f) = forms.(j) in
        zeros f.dtype (shape_of f))
  in
  let sums = starts st env forms ~at:nr s.seeds in
  let branch carried k =
    let b = s.branches.(k) in
    let fed = List.nth feeds k in
    let taken =
      Ops.where (Ops.ne cs.failed (int64 0)) no (Ops.eq index (at k))
    in
    (* The captures the branch reads are the call's parameters, so that its
       work on them runs only in its trip; another stands as a blank. A
       capture counts as read where any equation names it, a construct whose
       results go unread included, as its checks still run. *)
    let read = Array.mapi (fun j _ -> names b b.params.(j)) caps in
    let trip inner carried bound =
      let sums = Array.sub carried nr ns in
      let at = ref 0 in
      let caps =
        Array.mapi
          (fun j u ->
            if not read.(j) then blank u
            else begin
              incr at;
              bound.(!at - 1)
            end)
          caps
      in
      let out =
        outputs st inner ~reach (env_of b)
          (Array.append caps (Array.map (fun j -> sums.(j)) fed))
      in
      let base = Array.length out - Array.length fed in
      Array.iteri (fun m j -> sums.(j) <- out.(base + m)) fed;
      (Array.concat [ Array.sub out 0 nr; sums; [| no |] ], [||])
    in
    let carried, _ =
      repeat st cs ~n:1 ~reverse:false ~hoist:false
        ~stop:(Some (nr + ns))
        ~bound:(Loops.pick read caps)
        (Array.append carried [| taken |])
        [||] trip
    in
    Array.sub carried 0 (nr + ns)
  in
  let finals =
    List.fold_left branch (Array.append results sums) (List.init nb Fun.id)
  in
  Array.map (result env c) finals

and loop st cs env c =
  match c with
  | Iterate { rows = Some Taken; _ } ->
      decline "an iterate that stacks the steps it takes"
  | Scan { step; length = n; trips; reverse; init; xs; captures; seeds } ->
      let forms = Program.forms c in
      let nc = Array.length init and na = Array.length step.adds in
      let ny = Array.length step.outputs - nc - na in
      let _, _, feeds = Program.feeds step in
      let inits = Array.map (fun (V x) -> node st env x) init in
      let starts = starts st env forms ~at:(nc + ny) seeds in
      let finals =
        if n = 0 then
          Array.concat
            [
              inits;
              Array.init ny (fun j ->
                  let (F f) = forms.(nc + j) in
                  zeros f.dtype (shape_of f));
              starts;
            ]
        else
          let ns = Array.length starts in
          (* A scan that runs its first [t] rows is a loop that stops once it
             counts [t] trips, run forward over rows that a reversed scan
             reads reversed below [t]; its outputs are those rows, then zeros.
             [t] is an iterate's count of its steps, at most [n]. *)
          let t =
            Option.map
              (fun (V x) -> counted n (on_device st (node st env x)))
              trips
          in
          let xs =
            Array.map
              (fun (V x) ->
                let u = node st env x in
                match t with
                | Some t when reverse ->
                    leading (D.Any (Nx.dtype x)) ~reversed:true ~n t u
                | _ -> u)
              xs
          in
          let trip inner carried rows =
            let sub = env_of step in
            let caps = Array.map (fun (V x) -> node st env x) captures in
            let sums = Array.sub carried nc ns in
            (* The step's totals run from the sums they feed, which their
               running values out then are. *)
            let out =
              outputs st inner ~reach:(loop_reach env c) sub
                (Array.concat
                   [
                     Array.sub carried 0 nc;
                     rows;
                     caps;
                     Array.map (fun j -> sums.(j)) feeds;
                   ])
            in
            Array.iteri (fun k j -> sums.(j) <- out.(nc + ny + k)) feeds;
            let count =
              match t with
              | None -> [||]
              | Some t ->
                  let k = Ops.add carried.(nc + ns) (int32 1) in
                  [| k; Ops.lt k t |]
            in
            ( Array.concat [ Array.sub out 0 nc; sums; count ],
              Array.sub out nc ny )
          in
          let count =
            match t with
            | None -> [||]
            | Some t -> [| int32 0; Ops.lt (int32 0) t |]
          in
          let carried, ys =
            repeat st cs ~n
              ~reverse:(reverse && Option.is_none t)
              ~hoist:true
              ~stop:(Option.map (fun _ -> nc + ns + 1) t)
              ~bound:[||]
              (Array.concat [ inits; starts; count ])
              xs trip
          in
          let ys =
            match t with
            | None -> ys
            | Some t ->
                Array.mapi
                  (fun j u ->
                    let (F f) = forms.(nc + j) in
                    leading (D.Any f.dtype) ~reversed:reverse ~n t u)
                  ys
          in
          Array.concat [ Array.sub carried 0 nc; ys; Array.sub carried nc ns ]
      in
      Array.map (result env c) finals
  | Iterate
      {
        name;
        step;
        until;
        max;
        rows;
        stopped;
        init;
        step_captures;
        until_captures;
        seeds;
      } ->
      let forms = Program.forms c in
      let nc = Array.length init in
      let _, _, feeds = Program.sums [ until; step ] in
      let until_feeds = List.nth feeds 0 and step_feeds = List.nth feeds 1 in
      let ucaps = Array.map (fun (V x) -> node st env x) until_captures in
      let scaps = Array.map (fun (V x) -> node st env x) step_captures in
      (* The stop at a carry, adding its additions to the sums: whether the
         loop runs on, a boolean node, which no check of the program has
         failed by then. *)
      let reach = loop_reach env c in
      let test cs carry sums =
        let out =
          outputs st cs ~reach (env_of until)
            (Array.concat
               [ carry; ucaps; Array.map (fun j -> sums.(j)) until_feeds ])
        in
        let base = Array.length out - Array.length until_feeds in
        Array.iteri (fun k j -> sums.(j) <- out.(base + k)) until_feeds;
        let stop = Shape.reshape out.(0) (ints [ 1 ]) in
        let no = Ops.ne (int64 0) (int64 0) in
        Ops.where (Ops.ne cs.failed (int64 0)) no (Ops.eq stop (uint8 stop 0))
      in
      (* Where the iterate stacks its steps, it counts them in the carry, and
         each trip writes its outputs' row. *)
      let stacks = Option.is_some rows in
      let ny = Array.length step.outputs - nc - Array.length step.adds in
      let inits = Array.map (fun (V x) -> node st env x) init in
      let sums = starts st env forms ~at:(nc + Bool.to_int stacks + ny) seeds in
      let go = test cs inits sums in
      let ns = Array.length sums in
      let count = if stacks then [| int32 0 |] else [||] in
      let carried, ys =
        if max = 0 then
          ( Array.concat [ inits; sums; count; [| go |] ],
            Array.init ny (fun j ->
                let (F f) = forms.(nc + 1 + j) in
                zeros f.dtype (shape_of f)) )
        else
          let trip inner carried _ =
            let carry = Array.sub carried 0 nc in
            let sums = Array.sub carried nc ns in
            let out =
              outputs st inner ~reach (env_of step)
                (Array.concat
                   [ carry; scaps; Array.map (fun j -> sums.(j)) step_feeds ])
            in
            let carry = Array.sub out 0 nc in
            let base = Array.length out - Array.length step_feeds in
            Array.iteri (fun k j -> sums.(j) <- out.(base + k)) step_feeds;
            let go = test inner carry sums in
            let count =
              if stacks then [| Ops.add carried.(nc + ns) (int32 1) |] else [||]
            in
            (Array.concat [ carry; sums; count; [| go |] ], Array.sub out nc ny)
          in
          let carried, ys =
            repeat st cs ~n:max ~reverse:false ~hoist:true
              ~stop:(Some (nc + ns + Array.length count))
              ~bound:[||]
              (Array.concat [ inits; sums; count; [| go |] ])
              [||] trip
          in
          let ys =
            Array.mapi
              (fun j u ->
                let (F f) = forms.(nc + 1 + j) in
                leading (D.Any f.dtype) ~reversed:false ~n:max
                  carried.(nc + ns)
                  u)
              ys
          in
          (carried, ys)
      in
      (* Once the trips are done, the stop holds or the loop fails, naming
         the first lane still running where the lanes stop apart. *)
      let go = carried.(Array.length carried - 1) in
      let ok = Ops.where go (uint8 go 0) (uint8 go 1) in
      let fail lane =
        Invalid_argument
          (Printf.sprintf "%s: until is still false after max = %d steps%s"
             name max lane)
      in
      (match running forms carried stopped with
      | None -> raises cs ok (fun () -> fail "")
      | Some (lane, named) ->
          note cs
            {
              shape = [| 1 |];
              first = first_failure 1 (Shape.reshape ok (ints [ 1 ]));
              data = [ (D.Any D.Int64, lane) ];
              fail = (fun _ a -> fail (named a));
            });
      let count =
        if stacks then [| Shape.reshape carried.(nc + ns) [] |] else [||]
      in
      Array.map (result env c)
        (Array.concat
           [ Array.sub carried 0 nc; count; ys; Array.sub carried nc ns ])
  | c -> decline "%s" (construct_name c)

(* A loop's result reads every operand, nowhere at its own index, and so do
   its checks. *)
and loop_reach env c =
  let reads =
    List.fold_left
      (fun acc (V x) -> Is.union acc (reach env x).reads)
      Is.empty (operands c)
  in
  { reads; own = Is.empty }

and result env c u = (u, loop_reach env c, false)

(* Where each sum of a loop's additions starts: its seed, or the identity. *)
and starts st env forms ~at seeds =
  let n = Array.length forms - at in
  Array.init n (fun s ->
      if seeds = [||] then identity st forms.(at + s)
      else
        let (V x) = seeds.(s) in
        node st env x)

(* The nodes of [env]'s program's outputs, its parameters bound to [args]. *)
and outputs st cs ~reach env args =
  Array.iteri
    (fun k u ->
      env.nodes.(env.p.params.(k)) <- Some u;
      env.reaches.(env.p.params.(k)) <- reach)
    args;
  program st cs env;
  Array.map (fun (V y) -> node st env y) env.p.outputs

(* [repeat st cs ~n ~reverse ~hoist ~stop ~bound inits xs trip] is the values
   [n] trips of [trip] reach from [inits], and the outputs each trip stacks,
   computed on the program's device: [trip checks carried rows] maps the
   carried values and a row of each of [xs], stacked inputs of [n] rows, then
   each of [bound], to the next carried values and a trip's outputs, making its
   checks in [checks], and is applied once, to parameters of the call that a
   range of [n] trips runs. Trip [i] reads and writes row [i], or row
   [n - 1 - i] where [reverse] holds. Where [hoist] holds, the parts of the
   trip that read no parameter are computed once before the loop
   ({!Loop.cut}); otherwise they stay in the call, which a flag may skip. A
   value of [bound] is a parameter of the call, so that the trip's work on it
   stays there too. The checks are carried too, and become the loop's in
   [cs]. Where [stop] is [Some k], the carried value [k] is one boolean,
   whether to run on, which the engine reads before each trip. *)
and repeat st cs ~n ~reverse ~hoist ~stop ~bound inits xs trip =
  let device = st.device in
  let count = ref 0 in
  let next () =
    let k = !count in
    incr count;
    k
  in
  let own = Ops.Tbl.create 8 and args = ref [] and stores = ref [] in
  let pass slot u = args := (slot, u) :: !args in
  (* A parameter is made under a negative number no other parameter has, so
     that it stays apart from an enclosing body's parameters, which the step
     may read, and takes its slot once the body is cut. *)
  let held = ref [] in
  let unique () = -2 - Ops.unique_num () in
  let hold slot p =
    Ops.Tbl.replace own p ();
    held := (p, slot) :: !held;
    p
  in
  let range = Ops.range ~axis_type:Loop (Ops.Int n) [ Ops.unique_num () ] in
  let last = n - 1 in
  let row_at = if reverse then Ops.O.(int last - range) else range in
  let on_device u =
    if Option.is_none (Ops.device u) then Ops.copy_to_device u device else u
  in
  (* Each carried value: its buffer, holding its first value, the argument that
     stores it there, and its parameter's view. *)
  let carry u =
    if empty u then None
    else
      let slot = next () and k = Shape.max_numel u in
      let b = Ops.new_buffer device k (Ops.dtype u) in
      let p =
        hold slot
          (Tolk.Call.param ~shape:[ Ops.Int k ] ~device (unique ())
             (Ops.dtype u))
      in
      let first =
        let view = Shape.reshape b (Shape.shape u) in
        Ops.after b [ Ops.store view (on_device u) ]
      in
      pass slot first;
      Some (b, first, Shape.reshape p (Shape.shape u))
  in
  let view_of inits carried i =
    match carried.(i) with Some (_, _, v) -> v | None -> blank inits.(i)
  in
  let carried = Array.map carry inits in
  (* Each trip reads its row of each stacked input, from a copy whose rows are
     16 bytes apart where the input's are not. *)
  let rows =
    Array.map
      (fun x ->
        let u = on_device x in
        let row = List.tl (Shape.shape u) in
        let m = Shape.max_numel u / n in
        if m = 0 then blank (Shape.reshape u row)
        else
          let k = stride u m in
          let flat =
            if k = m then Shape.reshape u [ Ops.Int (n * m) ]
            else
              Shape.reshape
                (Shape.pad
                   (Shape.reshape u (ints [ n; m ]))
                   [ None; Some (Ops.Int 0, Ops.Int (k - m)) ])
                [ Ops.Int (n * k) ]
          in
          let w = window (Ops.contiguous flat) Ops.O.(row_at * int k) m in
          let slot = next () in
          let p = hold slot (Tolk.Call.param_like w (unique ())) in
          pass slot w;
          Shape.reshape p row)
      xs
  in
  let bound =
    Array.map
      (fun x ->
        let u = on_device x in
        if empty u then blank u
        else
          let flat =
            Ops.contiguous (Shape.reshape u [ Ops.Int (Shape.max_numel u) ])
          in
          let slot = next () in
          let p = hold slot (Tolk.Call.param_like flat (unique ())) in
          pass slot flat;
          Shape.reshape p (Shape.shape u))
      bound
  in
  let rows = Array.append rows bound in
  (* Whether a check of the program has failed by a trip's start: the
     enclosing one's at the loop's start, then each trip's at its end. *)
  let ncarried = Array.length inits in
  let flag_param =
    Tolk.Call.param ~shape:[ Ops.Int 1 ] ~device (unique ()) Int64
  in
  let before = Shape.reshape flag_param (ints [ 1 ]) in
  let inner = fresh_checks before in
  let nexts, ys =
    st.loops <- st.loops + 1;
    Fun.protect
      ~finally:(fun () -> st.loops <- st.loops - 1)
      (fun () -> trip inner (Array.init ncarried (view_of inits carried)) rows)
  in
  let made = List.rev inner.made in
  (* The flag is carried where the trip reads it or checks. *)
  let flag =
    let read = reads ~from:before in
    let reads_it u = read u <> Apart in
    let unread =
      not (Array.exists reads_it nexts || Array.exists reads_it ys)
    in
    if made = [] && unread then None
    else begin
      let slot = next () in
      ignore (hold slot flag_param);
      let b = Ops.new_buffer device 1 Int64 in
      let first = Ops.after b [ Ops.store b (on_device cs.failed) ] in
      pass slot first;
      Some (b, first, before)
    end
  in
  cs.read <- inner.read @ cs.read;
  (* The trip's checks, carried after its own values and the flag. *)
  let cinits = Array.of_list (check_inits made) in
  let ccarried = Array.map carry cinits in
  let cnexts =
    Array.of_list
      (check_nexts
         ~before:(Ops.ne before (int64 0))
         ~after:(Ops.ne inner.failed (int64 0))
         (Array.init (Array.length cinits) (view_of cinits ccarried))
         made)
  in
  let inits = Array.concat [ inits; [| cs.failed |]; cinits ] in
  let carried = Array.concat [ carried; [| flag |]; ccarried ] in
  let nexts = Array.concat [ nexts; [| inner.failed |]; cnexts ] in
  let view = view_of inits carried in
  let k = Array.length inits in
  (* A carried value whose next value reads another's in a cycle is copied,
     each trip, into storage of its own first. *)
  let how = Array.init k (fun i -> reads ~from:(view i)) in
  let edges = Array.make k [] and copied = Array.make k false in
  let seen = Array.make k false in
  let rec reaches j i =
    i = j
    || (not seen.(j))
       && begin
         seen.(j) <- true;
         List.exists (fun l -> reaches l i) edges.(j)
       end
  in
  for i = 0 to k - 1 do
    Array.fill seen 0 k false;
    let others =
      List.filter
        (fun j ->
          j <> i && Option.is_some carried.(j) && how.(j) nexts.(i) <> Apart)
        (List.init k Fun.id)
    in
    if List.exists (fun j -> reaches j i) others then copied.(i) <- true
    else edges.(i) <- others
  done;
  let finals =
    Array.mapi
      (fun i c ->
        match c with
        | None -> fun _ -> inits.(i)
        | Some (b, _, p) ->
            let u = inits.(i) and v = nexts.(i) in
            let v =
              if copied.(i) then begin
                let slot = next () and m = Shape.max_numel u in
                let w =
                  hold slot
                    (Tolk.Call.param ~shape:[ Ops.Int m ] ~device (unique ())
                       (Ops.dtype u))
                in
                pass slot (Ops.new_buffer device m (Ops.dtype u));
                let w = Shape.reshape w (Shape.shape u) in
                Ops.after w [ Ops.store w v ]
              end
              else if how.(i) v = Other then Ops.contiguous v
              else v
            in
            stores := Ops.store p v :: !stores;
            fun e -> Shape.reshape (Ops.after b [ e ]) (Shape.shape u))
      carried
  in
  (* Each trip writes its row of each stacked output. *)
  let ys =
    Array.map
      (fun u ->
        let dims = Ops.Int n :: Shape.shape u in
        let m = Shape.max_numel u in
        if m = 0 then fun _ -> blank (Shape.expand u dims)
        else
          let k = stride u m in
          let b = Ops.new_buffer device (n * k) (Ops.dtype u) in
          let w = window b Ops.O.(row_at * int k) m in
          let slot = next () in
          let p = hold slot (Tolk.Call.param_like w (unique ())) in
          pass slot w;
          stores :=
            Ops.store p (Shape.reshape (on_device u) [ Ops.Int m ]) :: !stores;
          fun e ->
            Shape.reshape
              (Shape.shrink
                 (Shape.reshape (Ops.after b [ e ]) (ints [ n; k ]))
                 [ None; Some (Ops.Int 0, Ops.Int m) ])
              dims)
      ys
  in
  let renumbered =
    List.map
      (fun (u, slot) ->
        match Ops.arg u with
        | Ops.Param a -> (u, Ops.replace u ~arg:(Ops.Param { a with slot }))
        | _ -> (u, u))
      !held
  in
  let body =
    Ops.substitute ~calls:Skip ~pass:Fixed_point
      (Loop.cut ~hoist ~own next args (Ops.sink (List.rev !stores)))
      renumbered
  in
  let by_slot (a, _) (b, _) = Int.compare a b in
  let call =
    Ops.call ~precompile:true body (List.map snd (List.sort by_slot !args))
  in
  (* A loop until a stop reads its flag, carried last, from the storage each
     trip updates. *)
  let e =
    match stop with
    | Some j -> (
        match carried.(j) with
        | Some (_, first, _) -> Ops.backedge call ~loop:range ~cond:first
        | None -> invalid_arg "Rune.jit: a loop's flag")
    | None -> Ops.end_ call [ range ]
  in
  let finals = Array.map (fun f -> f e) finals in
  if Option.is_some flag then
    cs.failed <- Ops.maximum cs.failed finals.(ncarried);
  check_finals cs made (Array.sub finals (ncarried + 1) (k - ncarried - 1));
  (Array.sub finals 0 ncarried, Array.map (fun f -> f e) ys)

(* [q], a program threading its totals, over [args]: its totals run from the
   sums of [c]'s results, each from its seed or the identity, which their
   running values out then are; its results, then the sums. *)
and threaded st cs env (q : Program.t) args seeds =
  threaded_over st cs env q (Array.map (operand st env) args) seeds

and threaded_over st cs env (q : Program.t) operands seeds =
  let _, sum_forms, feeds = Program.feeds q in
  let no = Array.length q.outputs - Array.length q.adds in
  let sums = starts st env sum_forms ~at:0 seeds in
  let sub = env_of q in
  Array.iteri
    (fun k (u, r, sc) ->
      let v = q.params.(k) in
      sub.nodes.(v) <- Some u;
      sub.reaches.(v) <- r;
      sub.scattered.(v) <- sc)
    operands;
  let na = Array.length operands in
  Array.iteri
    (fun k j ->
      let v = q.params.(na + k) in
      sub.nodes.(v) <- Some sums.(j);
      sub.reaches.(v) <- nothing;
      sub.scattered.(v) <- false)
    feeds;
  program st cs sub;
  let out =
    Array.map (fun (V y) -> (node st sub y, reach sub y, scattered_at sub y))
      q.outputs
  in
  let ends = Array.map (fun u -> (u, nothing, false)) sums in
  Array.iteri (fun k j -> ends.(j) <- out.(no + k)) feeds;
  Array.append (Array.sub out 0 no) ends

(* [q]'s outputs over the operands [args] of [env]'s program: each one's node,
   reach, and whether a scatter computes it. *)
and inline st cs env (q : Program.t) args =
  inline_over st cs q (Array.map (operand st env) args)

and inline_over st cs (q : Program.t) operands =
  let sub = env_of q in
  Array.iteri
    (fun k (u, r, sc) ->
      let v = q.params.(k) in
      sub.nodes.(v) <- Some u;
      sub.reaches.(v) <- r;
      sub.scattered.(v) <- sc)
    operands;
  program st cs sub;
  Array.map (operand st sub) q.outputs

(* [lower ~rig p forms ~layouts ~phases ~donated] is [p] over arguments of
   [forms], each laid out over its run as [layouts] says, its run at [phases],
   the arguments [donated] says lending their storage where a result may take
   it. *)
let lower ~rig ~uploads (p : Program.t) (forms : form array) ~layouts ~phases
    ~donated =
  let st =
    {
      rig;
      device = Ops.Single (Rig.name rig);
      captures = [];
      bound = [];
      uploads;
      views = Ops.Tbl.create 8;
      loops = 0;
    }
  in
  let env = env_of p in
  let params =
    Array.mapi
      (fun k (F f) ->
        let var = p.params.(k) in
        env.reaches.(var) <- { reads = Is.singleton k; own = Is.singleton k };
        let shape = shape_of f in
        if numel shape = 0 then begin
          env.nodes.(var) <- Some (zeros f.dtype shape);
          None
        end
        else begin
          on_rig st f.placement "an argument";
          let b, u =
            storage ~device:st.device ~phase:phases.(k) f.dtype layouts.(k)
          in
          env.nodes.(var) <- Some u;
          view_of st u;
          Some b
        end)
      forms
  in
  let cs = fresh_checks (int64 0) in
  program st cs env;
  (* Results lent their arguments' storage. *)
  let reaches = Array.map (fun (V y) -> reach env y) p.outputs in
  let checks_read = cs.read in
  let scattered_out = Array.map (fun (V y) -> scattered_at env y) p.outputs in
  let fits k j =
    let (F f) = forms.(k) in
    let (F g) = form p.outputs.(j) in
    donated.(k)
    && Option.is_some params.(k)
    && (not (List.exists (fun r -> Is.mem k r.reads) checks_read))
    && Nx.Dtype.equal f.dtype g.dtype
    && numel (shape_of f) = numel (shape_of g)
    && Nx.Placement.equal f.placement g.placement
    && L.is_contiguous layouts.(k)
    && L.offset layouts.(k) = 0
    && phases.(k) = 0
  in
  let computed = Array.map (fun (V y) -> var env y >= 0) p.outputs in
  (* Arguments are lent, and lent again without each pair whose stores close a
     cycle, until the stores have an order. *)
  let dtype j =
    let (F g) = form p.outputs.(j) in
    D.Any g.dtype
  in
  let buffer k = Option.get params.(k) in
  let store_over j k u =
    let (D.Any dt) = dtype j in
    store dt (buffer k) u
  in
  let view j s =
    let (D.Any dt) = dtype j in
    let (F g) = form p.outputs.(j) in
    read_back dt (shape_of g) s
  in
  let rec settle refused =
    let fits k j = fits k j && not (List.mem (k, j) refused) in
    let scattered = scattered_out in
    let lent = lend ~donated ~computed ~reaches ~scattered ~fits in
    let nodes =
      Array.mapi
        (fun j (V y) ->
          match var env y with
          | v when v >= 0 && not (lent.(j) >= 0 && v = p.params.(lent.(j))) ->
              Some (node st env y)
          | _ -> None)
        p.outputs
    in
    let checks =
      List.map (fun (c : made) -> c.first :: List.map snd c.data) cs.made
    in
    match assign ~nodes ~checks ~store:store_over ~view ~buffer lent with
    | Ok a -> (lent, a)
    | Error j -> settle ((lent.(j), j) :: refused)
  in
  let lent, assigned = settle [] in
  let stores = ref [] in
  let buffer dt n =
    if packed dt then Ops.new_buffer st.device (D.bytes dt n) Uint8
    else Ops.new_buffer st.device (n * parts dt) (tdtype dt)
  in
  let stored (F f) u =
    let n = numel (shape_of f) in
    if n = 0 then Empty
    else
      let o = buffer f.dtype n in
      stores := store f.dtype o u :: !stores;
      Stored o
  in
  let outs =
    Array.mapi
      (fun j (V y) ->
        let f = Nx.Prim.form y in
        match var env y with
        | v when v >= 0 -> (
            on_rig st f.placement "a result";
            match lent.(j) with
            | k when k >= 0 && v = p.params.(k) -> Taken k
            | k when k >= 0 ->
                stores := Option.get assigned.stored.(j) :: !stores;
                Lent k
            | _ -> stored (F f) (assigned.read j (node st env y)))
        | _ -> (
            match elsewhere st y with
            | Some a -> Copied a
            | None ->
                on_rig st f.placement "a result";
                stored (F f) (capture st y)))
      p.outputs
  in
  let one dt u =
    let o = buffer dt 1 in
    stores := store dt o u :: !stores;
    o
  in
  let checks =
    List.rev_map
      (fun (c : made) ->
        let read = assigned.read (-1) in
        let first = one D.Int64 (read c.first) in
        let data =
          List.map (fun ((D.Any dt as d), u) -> (d, one dt (read u))) c.data
        in
        ({ shape = c.shape; first; data; fail = c.fail } : check))
      cs.made
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
   slots of the first one timed, compiled on [parallel] domains, or as many as
   the [PARALLEL] setting gives. *)
let search rig ~parallel width k =
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
  let search () =
    Tolk.Search.beam_search ~link ~time ~clock:Engine.clock width k
  in
  match parallel with
  | None -> search ()
  | Some p -> Tolk.Setting.context [ B (Tolk.Setting.parallel, p) ] search

(* The settings a caller may change around a call that change what tolk
   compiles: each setting that shapes it, and whether the engine links each
   kernel as a submission of its own, as it does for its reports. *)
type settings = (string * string) list * bool

(* [shaping beam] is tolk's settings that shape a program lowered at the width
   [beam], or at the settings' ({!Tolk.Jit.shaping}). It is made again only
   when tolk's settings change, so that reading it, as a call does to find its
   variant, allocates no list. *)
let shaping beam =
  let last = Atomic.make ([], []) in
  fun () ->
    let entries = Tolk.Setting.shaping () in
    let seen, shaped = Atomic.get last in
    if seen == entries then shaped
    else
      let shaped = Tolk.Jit.shaping ?beam () in
      Atomic.set last (entries, shaped);
      shaped

let settings shaping =
  (shaping (), Tolk.Setting.value Tolk.Setting.debug >= 2)

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

let link ~beam ~parallel rig l =
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
          (function
            | Stored o -> slot o | Lent _ | Taken _ | Copied _ | Empty -> -1)
          l.outs
      in
      let checks =
        List.map
          (fun (c : check) ->
            (c, slot c.first, List.map (fun (_, o) -> slot o) c.data))
          l.checks
      in
      let linear, vars =
        Tolk.Schedule.create_linear_with_vars ~capturing:true sink
      in
      let linear =
        Tolk.Jit.jit_lower ?beam ~search:(search rig ~parallel)
          ~targets:(fun _ -> Engine.target rig)
          ~held_bufs:(List.map fst l.bound) ~inputs:(List.rev !order) linear
      in
      let schedule =
        Engine.link ~devices:(fun _ -> rig) ~bound:l.bound linear
      in
      {
        schedule = Some (schedule, vars);
        inputs = !next;
        leaf_slots;
        out_slots;
        outs = l.outs;
        checks;
      }

(* Programs *)

(* An output's form, and for a placed one the layout and bytes of the memory
   its result takes. *)
type output = { form : form; contiguous : L.t; bytes : int }

let output_of (F f as form) =
  let shape = shape_of f in
  {
    form;
    contiguous = L.contiguous (Iarray.of_array shape);
    bytes = D.bytes f.dtype (numel shape);
  }

type t = {
  name : string;
  rig : Rig.t;
  program : Program.t;
  forms : form array;
  results : output array;  (** Each output's, read once. *)
  beam : int option;  (** The search width the caller gave. *)
  parallel : int option;
  shaping : unit -> (string * string) list;  (** {!shaping} at [beam]. *)
  lock : Mutex.t;
  variants : variant list Atomic.t;
  uploads : (value * A.any) list ref;
      (** The copies of the constants of other devices, made by the first
          lowering, under [lock]. *)
  mutable first : (L.t array * lowered) option;
      (** The lowering {!make} checks, at contiguous layouts, phase [0] and no
          donation, until the first variant takes it, under [lock]: a program's
          first call lowers it once. *)
}

(* The program's device: the one its first argument, or else its first result,
   lies whole on. *)
let device (p : Program.t) forms =
  let first = ref None in
  let consider (F f) =
    match !first with
    | None -> (
        match device_of f.placement with
        | Some r -> first := Some r
        | None ->
            decline "a value placed at %s, over several devices"
              (Format.asprintf "%a" Nx.Placement.pp f.placement))
    | Some _ -> ()
  in
  Array.iter consider forms;
  Array.iter (fun v -> consider (form v)) p.outputs;
  match !first with
  | Some r when Rig.computes r -> r
  | Some r -> decline "a value on %s, which computes nothing" (Rig.name r)
  | None -> decline "no value"

(* What a program the compiler takes must not raise to compile: an operation it
   does not lower, or tolk refusing the graph, as a target whose compiler is
   missing does. *)
let declined = function
  | Declined why -> Some why
  | Invalid_argument why | Failure why -> Some why
  | _ -> None

(* Whether two settings are equal: [Tolk.Setting.shaping] gives one list while
   no setting changes, so equal lists are mostly one. *)
let same_settings ((s, d) : settings) ((s', d') : settings) =
  Bool.equal d d' && (s == s' || s = s')

let same_variant layouts phases donated settings v =
  Array.for_all2 Int.equal v.phases phases
  && Array.for_all2 Bool.equal v.donated donated
  && same_settings v.settings settings
  && Array.for_all2 L.equal v.layouts layouts

let variants_made = Domain.DLS.new_key (fun () -> ref 0)
let variants () = !(Domain.DLS.get variants_made)

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
          let first = t.first in
          t.first <- None;
          let lowered () =
            match first with
            | Some (l, lowered)
              when Array.for_all2 L.equal l layouts
                   && Array.for_all (Int.equal 0) phases
                   && not (Array.exists Fun.id donated) ->
                lowered
            | _ ->
                lower ~rig:t.rig ~uploads:t.uploads t.program t.forms ~layouts
                  ~phases ~donated
          in
          (* A declined variant is kept; another failure, as an interrupted
             search, keeps nothing. *)
          let linked =
            match
              link ~beam:t.beam ~parallel:t.parallel t.rig (lowered ())
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
          incr (Domain.DLS.get variants_made);
          v)

let make ~name ?beam ?parallel p forms =
  match
    let rig = device p forms in
    let layouts =
      Array.map
        (fun (F f) -> within f.dtype (L.contiguous (L.shape f.layout)))
        forms
    in
    let phases = Array.map (fun _ -> 0) forms in
    let donated = Array.map (fun _ -> false) forms in
    let uploads = ref [] in
    let lowered = lower ~rig ~uploads p forms ~layouts ~phases ~donated in
    {
      name;
      rig;
      program = p;
      forms;
      results = Array.map (fun v -> output_of (form v)) p.outputs;
      beam;
      parallel;
      shaping = shaping beam;
      lock = Mutex.create ();
      variants = Atomic.make [];
      uploads;
      first = Some (layouts, lowered);
    }
  with
  | t -> Ok t
  | exception e -> (
      match declined e with Some why -> Error why | None -> raise e)

(* Calls *)

let evaluations_made = Domain.DLS.new_key (fun () -> ref 0)
let evaluations () = !(Domain.DLS.get evaluations_made)
let evaluated_call () = incr (Domain.DLS.get evaluations_made)

(* [consumed ~by args f] is [f arrays take] under a consumer of [args],
   [arrays k] the arrays of argument [k] and [take k a] the memory of its array
   [a] for a result, if it may take it ({!Nx.Prim.Consumer.take}). *)
let consumed ~by args f =
  match List.init (Array.length args) Fun.id with
  | [] ->
      let none _ = invalid_arg (by ^ ": no argument") in
      f none (fun k _ -> none k)
  | k0 :: _ as all ->
      let (V x0) = args.(k0) in
      let set = Nx.Placement.devices (Nx.placement x0) in
      let any k =
        let (V y) = args.(k) in
        match Nx.Prim.coerce (Nx.dtype y) set y with
        | Some y -> Nx.Prim.Any y
        | None -> invalid_arg (by ^ ": arguments on several sets")
      in
      Nx.Prim.Consumer.with_ ~by (List.map any all) (fun c ->
          f
            (fun k -> Nx.Prim.Consumer.arrays c k)
            (fun k (A.Any a) ->
              Option.map A.buffer
                (Nx.Prim.Consumer.take c k (A.dtype a) (A.layout a))))

let value rig r buffer =
  let (F f) = r.form in
  let buffer =
    match buffer with Some b -> b | None -> Rig.Buffer.create rig r.bytes
  in
  V
    (Nx.Repr.unsafe_of_array (Nx.Placement.devices f.placement)
       (A.v f.dtype r.contiguous buffer))

(* Evaluation *)

(* [x] as a value over [arrays], its arrays under a consumer. *)
let over (V x) arrays =
  let f = Nx.Prim.form x in
  V
    (Nx.Repr.of_shards f.placement
       (Iarray.of_array (Array.map (A.expect f.dtype) arrays)))

(* [p] over [args], its arguments then its totals' running values in. *)
let eval_all (p : Program.t) args =
  let n = Array.length args - Array.length p.adds in
  eval ~totals:(Array.sub args n (Array.length p.adds)) p (Array.sub args 0 n)

(* [p] evaluated over the arrays of a consumer of [args]: the donors it claimed
   are dead, so the arrays of every argument stand for it, a donor beside
   its handle included. Each result is copied, since no value over the arrays
   may outlive the consumer. A value over an argument's memory shares it for
   good, so a later donation of that argument computes into fresh memory: the
   cost of this path. *)
let evaluated p args arrays =
  let values = Array.mapi (fun k v -> over v (arrays k)) args in
  (* A callback may keep what it receives, so it receives copies. *)
  let values =
    if Program.holds (function Callback _ -> true | _ -> false) p then
      Array.map (fun (V x) -> V (Nx.copy x)) values
    else values
  in
  Array.map (fun (V y) -> V (Nx.copy y)) (eval_all p values)

let evaluate ~name p args =
  if Array.exists (fun (V x) -> Nx.Prim.donated x) args then
    consumed ~by:name args (fun arrays _ -> evaluated p args arrays)
  else
    (* Each result is a copy. [eval] returns an argument or a constant as
       itself, and a reshape of one lies over its memory. *)
    Array.map (fun (V y) -> V (Nx.copy y)) (eval_all p args)

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
  match (variant t layouts phases donated (settings t.shaping)).linked with
  | Error _ ->
      evaluated_call ();
      evaluated p args arrays
  | Ok v -> linked t v args leaves take

(* [linked t v args leaves take] runs the variant [v] of [t] on [args], whose
   arrays under the call's consumer are [leaves]. *)
and linked t v args leaves take =
  let slots = Array.make v.inputs [] in
  (* The run an argument binds: its buffer itself where the run is all of
     it. *)
  let run_of k =
    let (A.Any a) = leaves.(k) in
    let first, length = run_bytes (A.dtype a) (A.layout a) in
    let b = A.buffer a in
    if first = 0 && length = Rig.Buffer.length b then b
    else Rig.Buffer.view b ~first ~length
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
  let buffers =
    Array.mapi
      (fun j s ->
        match v.outs.(j) with
        | Lent k | Taken k -> lending.(k)
        | Stored _ ->
            let b = Rig.Buffer.create t.rig t.results.(j).bytes in
            slots.(s) <- [ b ];
            Some b
        | Copied src ->
            (* Queued: the stamps order a later access of [dst] after it. *)
            let length = Rig.Buffer.length src in
            let dst = Rig.Buffer.create (Rig.Buffer.device src) length in
            if length > 0 then
              ignore
                (Rig.Buffer.submit_copy ~src ~dst [| Rig.Buffer.row length |]);
            Some dst
        | Empty -> None)
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
  Array.mapi (fun j r -> value t.rig r buffers.(j)) t.results
