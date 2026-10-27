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
   that has no address the host computes on is laid out by its device, which
   starts an allocation on its alignment. *)
let phase b first =
  let rec find j =
    if j >= alignment || Rig.Buffer.aligned b ~at:(first - j) alignment then j
    else find (j + 1)
  in
  match find 0 with j -> j | exception Invalid_argument _ -> 0

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

(* The layout a parameter of form [f] binds: its own, or for a value of
   every set, which a call computes afresh, C order. *)
let layout (F f) =
  match f.placement with
  | Some _ -> f.layout
  | None -> L.contiguous (L.shape f.layout)

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
  | Empty  (** No element. *)
  | Kept of value  (** A constant of every set, returned as it is. *)

type lowered = {
  sink : Ops.t option;  (** [None] where no result is stored. *)
  params : Ops.t option array;  (** Each argument's buffer, if it has one. *)
  outs : out array;
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

let lower ~rig (p : Program.t) (forms : form array) phases =
  let st = { rig; device = Ops.Single (Rig.name rig); captures = []; bound = [] } in
  let env = Array.make p.slots None in
  let params =
    Array.mapi
      (fun k (F f) ->
        let var = p.params.(k) in
        let shape = shape_of f in
        if numel shape = 0 then begin
          env.(var) <- Some (zeros f.dtype shape);
          None
        end
        else begin
          Option.iter (fun q -> on_rig st q "an argument") f.placement;
          let b, u =
            storage ~device:st.device ~phase:phases.(k) f.dtype (layout (F f))
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
  let issue (type r) base (op : r Nx.Prim.t) =
    (match op with
    | Place (q, _) -> on_rig st q "a placement"
    | Check _ -> decline "a check"
    | _ -> ());
    let ys =
      match Lower.op node op with
      | ys -> ys
      | exception Invalid_argument why -> decline "%s" why
    in
    Array.iteri (fun k y -> env.(base + k) <- Some y) ys
  in
  Array.iter
    (function
      | Op { op; base; _ } -> issue base op
      | Do { c = Detach x; base } -> env.(base) <- Some (node x)
      | Do { c; _ } -> decline "%s" (construct c))
    p.eqns;
  let stores = ref [] in
  let stored (F f) u =
    let shape = shape_of f in
    if numel shape = 0 then Empty
    else
      let n = numel shape in
      let o =
        if packed f.dtype then
          Ops.new_buffer st.device (D.bytes f.dtype n) Uint8
        else Ops.new_buffer st.device (n * parts f.dtype) (tdtype f.dtype)
      in
      stores := store f.dtype o u :: !stores;
      Stored o
  in
  let outs =
    Array.map
      (fun (V y) ->
        let f = Nx.Prim.form y in
        match Nx.Prim.payload p.owner y with
        | Some (Var _) -> (
            match f.placement with
            | None -> decline "a result of every set"
            | Some q ->
                on_rig st q "a result";
                stored (F f) (node y))
        | _ when Option.is_none f.placement -> Kept (V y)
        | _ -> stored (F f) (capture st y))
      p.outputs
  in
  let sink =
    match !stores with [] -> None | s -> Some (Ops.sink (List.rev s))
  in
  { sink; params; outs; bound = st.bound }

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

(* A program linked for one phase of each argument: the linked schedule, its
   variables, and the slot of each argument and output, [-1] for none. *)
type variant = {
  phases : int array;
  linked : (Engine.t * (string * int) list) option;
  inputs : int;
  leaf_slots : int array;
  out_slots : int array;
}

let link rig l phases =
  match l.sink with
  | None ->
      {
        phases;
        linked = None;
        inputs = 0;
        leaf_slots = Array.map (fun _ -> -1) l.params;
        out_slots = Array.map (fun _ -> -1) l.outs;
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
        Array.map (function Stored o -> slot o | Empty | Kept _ -> -1) l.outs
      in
      let linear, vars =
        Tolk.Schedule.create_linear_with_vars ~capturing:true sink
      in
      let linear =
        Tolk.Jit.jit_lower ~search:(search rig)
          ~targets:(fun _ -> Engine.target rig)
          ~held_bufs:(List.map fst l.bound) ~inputs:(List.rev !order) linear
      in
      let linked = Engine.link ~devices:(fun _ -> rig) ~bound:l.bound linear in
      { phases; linked = Some (linked, vars); inputs = !next; leaf_slots; out_slots }

(* Programs *)

type t = {
  name : string;
  rig : Rig.t;
  program : Program.t;
  forms : form array;
  lock : Mutex.t;
  variants : variant list Atomic.t;
}

(* The program's device: its arguments', or else its results'. *)
let device (p : Program.t) forms =
  let first = ref None in
  let consider (F f) =
    match (!first, f.placement) with
    | None, Some q -> (
        match device_of q with
        | Some r -> first := Some r
        | None ->
            decline "a value placed at %s, over several devices"
              (Format.asprintf "%a" Nx.Placement.pp q))
    | _ -> ()
  in
  Array.iter consider forms;
  Array.iter (fun v -> consider (form v)) p.outputs;
  match !first with
  | Some r when Rig.computes r -> r
  | Some r -> decline "a value on %s, which computes nothing" (Rig.name r)
  | None -> decline "no value placed"

let make ~name p forms =
  match
    let rig = device p forms in
    let zeros = Array.map (fun _ -> 0) forms in
    ignore (lower ~rig p forms zeros);
    rig
  with
  | rig ->
      Ok
        {
          name;
          rig;
          program = p;
          forms;
          lock = Mutex.create ();
          variants = Atomic.make [];
        }
  | exception Declined why -> Error why

let variant t phases =
  let find () =
    List.find_opt (fun v -> v.phases = phases) (Atomic.get t.variants)
  in
  match find () with
  | Some v -> v
  | None -> (
      Mutex.protect t.lock @@ fun () ->
      match find () with
      | Some v -> v
      | None ->
          let v = link t.rig (lower ~rig:t.rig t.program t.forms phases) phases in
          Atomic.set t.variants (v :: Atomic.get t.variants);
          v)

(* Calls *)

(* [consumed ~by args f] is [f arrays] under a consumer of the placed values of
   [args], [arrays k] the arrays of argument [k], placed. *)
let consumed ~by args f =
  let placed =
    List.filter
      (fun k -> match args.(k) with V x -> Option.is_some (Nx.placement x))
      (List.init (Array.length args) Fun.id)
  in
  match placed with
  | [] -> f (fun _ -> invalid_arg (by ^ ": no argument is placed"))
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
          f (fun k -> Nx.Prim.Consumer.arrays c position.(k)))

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

let run t args =
  let p = t.program in
  consumed ~by:t.name args @@ fun arrays ->
  (* Each argument's array: its own, or a value of every set computed. *)
  let array k =
    let (V x) = args.(k) in
    match Nx.placement x with
    | None -> computed t.rig x
    | Some _ -> (arrays k).(0)
  in
  let leaves = Array.init (Array.length args) array in
  let phases =
    Array.map
      (fun (A.Any a) ->
        let l = A.layout a in
        if L.numel l = 0 then 0
        else phase (A.buffer a) (fst (run_bytes (A.dtype a) l)))
      leaves
  in
  let v = variant t phases in
  let slots = Array.make v.inputs [] in
  Array.iteri
    (fun k s ->
      if s >= 0 then
        let (A.Any a) = leaves.(k) in
        let first, length = run_bytes (A.dtype a) (A.layout a) in
        slots.(s) <- [ Rig.Buffer.view (A.buffer a) ~first ~length ])
    v.leaf_slots;
  let outputs = Array.map form p.outputs in
  let buffers =
    Array.mapi
      (fun j s ->
        if s < 0 then None
        else
          let (F f) = outputs.(j) in
          let b =
            Rig.Buffer.create t.rig (D.bytes f.dtype (numel (shape_of f)))
          in
          slots.(s) <- [ b ];
          Some b)
      v.out_slots
  in
  Option.iter (fun (l, vars) -> Engine.run ~vars l slots) v.linked;
  (* A result of every set is a constant: the program declines one it
     computes. *)
  Array.mapi
    (fun j (F f as o) ->
      match f.placement with
      | None -> p.outputs.(j)
      | Some _ -> result t.rig o buffers.(j))
    outputs
