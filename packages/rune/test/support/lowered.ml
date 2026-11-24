(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module A = Nx_array
module D = Nx_array.Dtype
module L = Rune.Lower_arith

type ('v, 's, 'd) Nx.Prim.payload +=
  | Node : Ops.t -> ('v, 's, 'd) Nx.Prim.payload

let cpu = "CPU"

(* Where a graph computes: its device, the renderer of its kernels and
   whether they compile for the host's processors. *)
type place = { device : string; renderer : Renderer.t Lazy.t; host : bool }

(* The host's target, or skips the running test where the host's programs cannot
   compile ({!Tolk_engine.compiles}). *)
let host_target () =
  match Tolk_engine.compiles Rig.host with
  | Ok () -> Tolk_engine.target Rig.host
  | Error reason -> Windtrap.skip ~reason ()

let on_host =
  let renderer =
    lazy
      (ignore (host_target ());
       Tolk_engine.renderer Rig.host)
  in
  { device = cpu; renderer; host = true }
let numel s = Array.fold_left ( * ) 1 s

(* The elements' bytes as a node holds them: one element of [L.dtype] per
   element, two for a complex one. *)
let parts (type v s) (dt : (v, s) D.t) =
  match D.kind dt with D.Complex -> 2 | _ -> 1

let host_buffer bytes =
  let b = Rig.Buffer.create Rig.host (max 1 (String.length bytes)) in
  Rig.Buffer.blit_from_string bytes 0 b 0 (String.length bytes);
  b

(* Storage

   A capture reads its array's storage through its layout, as a gather of the
   storage's elements at offset + Σ index·stride, when its dtype's storage holds
   one element per node element. A sub-byte dtype's storage packs its elements,
   so it is bound by its elements, in C order. *)

let packed (type v s) (dt : (v, s) D.t) = D.bits dt < 8

let elements (type v s) (dt : (v, s) D.t) (xs : v array) =
  let element : v -> string =
    match dt with
    | D.Int4 -> fun x -> String.make 1 (Char.chr (x land 0xff))
    | _ -> Nx_kernel.Prog.bits dt
  in
  String.concat "" (Array.to_list (Array.map element xs))

let strided a flat =
  let l = A.layout a in
  let shape = Iarray.to_array (A.Layout.shape l) in
  let r = Array.length shape in
  let at =
    Array.to_list shape
    |> List.mapi (fun i n ->
        let step = A.Layout.stride l i in
        let v = Shape.arange ~dtype:Weak_int n in
        let v = if step = 1 then v else Ops.mul v (Ops.int step) in
        Shape.expand
          (Shape.reshape v
             (List.init r (fun j -> Ops.Int (if j = i then n else 1))))
          (List.map (fun n -> Ops.Int n) (Array.to_list shape)))
    |> List.fold_left Ops.add
         (Shape.const_like ~dtype:Weak_int
            (Shape.expand (Ops.int 0)
               (List.map (fun n -> Ops.Int n) (Array.to_list shape)))
            (`Int (Bigint.of_int (A.Layout.offset l))))
  in
  Ops.index flat [ at ]

(* [bind ~direct bound x] binds [x]'s storage; where [direct], an array laid
   out contiguously from its storage's start is read as the storage itself,
   as a compiled call reads its argument. *)
let bind (type v s d) ~direct ~device bound (x : (v, s, d) Nx.t) =
  let dt = Nx.dtype x and shape = Nx.shape x in
  let tdt = L.dtype (D.Any dt) and k = parts dt in
  let leaf bytes n =
    let u = Ops.new_buffer device (max 1 n) tdt in
    bound := (u, bytes) :: !bound;
    u
  in
  match Nx.Repr.array ~by:"test" x with
  | Some a when (not (packed dt)) && numel shape > 0 ->
      let b = A.buffer a in
      let len = Rig.Buffer.length b in
      let bytes = Bytes.create len in
      Rig.Buffer.blit_to_bytes b 0 bytes 0 len;
      let n = len / Dtype.itemsize tdt in
      let flat = leaf (Bytes.to_string bytes) n in
      let flat =
        if k = 1 then flat
        else Shape.reshape flat [ Ops.Int (n / k); Ops.Int k ]
      in
      let l = A.layout a in
      if direct && A.Layout.is_contiguous l && A.Layout.offset l = 0
         && numel shape * k = n
      then Shape.reshape flat (L.dims (D.Any dt) shape)
      else strided a flat
  | _ when numel shape = 0 ->
      let dims = L.dims (D.Any dt) shape in
      Shape.expand
        (Shape.reshape
           (Ops.const ~dtype:tdt (`Int Bigint.zero))
           (List.map (fun _ -> Ops.Int 1) dims))
        dims
  | _ ->
      let xs = Nx.to_array x in
      let n = Array.length xs * k in
      Shape.reshape (leaf (elements dt xs) n) (L.dims (D.Any dt) shape)

(* Reading back *)

let decode (type v s) (dt : (v, s) D.t) shape (bytes : Bytes.t) : v array =
  let n = numel shape in
  let byte i = Char.code (Bytes.get bytes i) in
  match dt with
  | D.Int4 ->
      Array.init n (fun i ->
          let c = byte i in
          if c >= 128 then c - 256 else c)
  | D.Uint4 -> Array.init n byte
  | D.Bit -> Array.init n (fun i -> byte i <> 0)
  | D.Float4_e2m1fn ->
      let magnitude = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |] in
      Array.init n (fun i ->
          let c = byte i in
          let m = magnitude.(c land 7) in
          if c land 8 <> 0 then -.m else m)
  | _ ->
      let b = host_buffer (Bytes.to_string bytes) in
      A.to_array (A.v dt (A.Layout.contiguous (Iarray.of_array shape)) b)

(* Lowering

   A value [f] did not compute is bound once, however many operations read it,
   as a compiled call binds each argument and each value it captures: by
   identity. *)

type value = V : ('v, 's, 'd) Nx.t -> value

let same (type v s d) (x : (v, s, d) Nx.t) (V y) =
  let set = Nx.Placement.devices (Nx.placement x) in
  match Nx.Prim.coerce (Nx.dtype x) set y with
  | Some y -> y == x
  | None -> false

let rule ~direct ~local ~place ?transposed bound seen i ~by (type r)
    (op : r Nx.Prim.t) : r =
  let node : type v s d. (v, s, d) Nx.t -> Ops.t =
   fun x ->
    match Nx.Prim.payload i x with
    | Some (Node u) -> u
    | Some _ -> invalid_arg "Lowered: a value of another payload"
    | None -> (
        match List.find_opt (fun (y, _) -> same x y) !seen with
        | Some (_, u) -> u
        | None ->
            let u = bind ~direct ~device:place.device bound x in
            seen := (V x, u) :: !seen;
            u)
  in
  (match (transposed, op) with
  | Some found, Nx.Prim.Attention _ ->
      found :=
        !found
        @ [
            Rune.attention_transposed ~device:place.device
              ~renderer:(Lazy.force place.renderer) node op;
          ]
  | _ -> ());
  let ys =
    Rune.Lower.op ~renderer:place.renderer ~device:place.device
      ~host:place.host ~local node op
  in
  Nx.Prim.results ~by (fun k f -> Nx.Prim.traced i f (Node ys.(k))) op

(* A result of [f ()] lowered: its node, dtype and shape, and the count of its
   node's elements. *)
type ('v, 's) out = {
  node : Ops.t;
  dt : ('v, 's) D.t;
  shape : int array;
  n : int;
}

let out (type v s d) i (y : (v, s, d) Nx.t) : (v, s) out =
  match Nx.Prim.payload i y with
  | Some (Node node) ->
      let dt = (Nx.Prim.form y).dtype and shape = Nx.shape y in
      { node; dt; shape; n = numel shape * parts dt }
  | _ -> invalid_arg "Lowered: the result is not lowered"

(* [f i]'s graph, whose results [f] reads with [out i], and the storage it
   binds. *)
let lowered ?(direct = false) ?(local = true) ?(place = on_host) ?transposed f =
  let bound = ref [] and seen = ref [] in
  let r =
    Nx.Prim.interpret ~name:"Lowered" Nx.Prim.Extent
      (fun i ~by op ->
        rule ~direct ~local ~place ?transposed bound seen i ~by op)
      f
  in
  (!bound, r)

let one f i = out i (f ())

(* The host's tensor cores, as its renderer's kernels take them. *)
let host_cores () = [ (cpu, Heuristic.cores (Lazy.force on_host.renderer)) ]

(* The sink of the stores of each graph [u] of [stores], of [n] elements, into
   its buffer [o]. *)
let sink stores =
  Ops.sink
    (List.map
       (fun (o, u, _, n) ->
         Ops.after o [ Ops.store o (Shape.reshape u [ Ops.Int n ]) ])
       stores)

(* The store of [o]'s graph, of at least one element, into a buffer of its own
   on [device]. *)
let stored (type v s) ?(device = cpu) (o : (v, s) out) =
  let n = max 1 o.n and tdt = L.dtype (D.Any o.dt) in
  sink [ (Ops.new_buffer device n tdt, o.node, tdt, n) ]

(* The schedule of [sink], each device taking the tensor cores [cores] gives it
   ({!Schedule}), the host's own by default, and {!compiled} it compiled. *)
let schedule ?(cores = host_cores ()) sink =
  Schedule.create_linear_with_vars ~cores sink

let compiled ?cores sink =
  let linear, vars = schedule ?cores sink in
  let targets _ = host_target () in
  (Realize.compile_linear ~targets linear, vars)

(* Programs

   A graph is compiled once: a graph equal to one compiled before but for the
   slots of its storage runs that one's program, its storage's bytes copied into
   the program's, as a compiled call runs one variant over each call's
   arguments. Storage nodes are numbered in the order a walk of the graph
   reaches them, which two such graphs share, and a program keeps each buffer it
   binds by its node's number. The key holds the settings the kernels compile
   under. *)

type program = {
  linked : Tolk_engine.t;
  vars : (string * int) list;
  storage : Rig.Buffer.t option array;
  results : Rig.Buffer.t list;
}

let programs : (string, program) Hashtbl.t = Hashtbl.create 64

let numbered sink =
  let numbers = Ops.Tbl.create 16 in
  List.iter
    (fun u ->
      if Op.equal (Ops.op u) Op.Buffer && not (Ops.Tbl.mem numbers u) then
        Ops.Tbl.add numbers u (Ops.Tbl.length numbers))
    (Ops.toposort ~calls:Enter sink);
  numbers

let program_key numbers sink =
  let arg u =
    Option.map
      (fun slot -> Ops.Param { (Ops.param_arg_of u) with slot })
      (Ops.Tbl.find_opt numbers u)
  in
  let setting (name, value) = name ^ "=" ^ value in
  String.concat "\n"
    (Ops.key_with ~arg sink :: List.map setting (Setting.shaping ()))

(* [sink] compiled, its stores' buffers made and [bound]'s storage bound. *)
let link ?cores numbers sink bound stores =
  let compiled, vars = compiled ?cores sink in
  let storage = Array.make (Ops.Tbl.length numbers) None in
  let inputs =
    List.map
      (fun (u, bytes) ->
        let b = host_buffer bytes in
        Option.iter
          (fun i -> storage.(i) <- Some b)
          (Ops.Tbl.find_opt numbers u);
        (u, b))
      bound
  in
  let results =
    List.map
      (fun (o, _, tdt, n) ->
        (o, Rig.Buffer.create Rig.host (n * Dtype.itemsize tdt)))
      stores
  in
  let linked =
    Tolk_engine.link
      ~devices:(fun _ -> Rig.host)
      ~bound:(results @ inputs) compiled
  in
  { linked; vars; storage; results = List.map snd results }

(* [bound]'s bytes copied into [p]'s storage. *)
let refill p numbers bound =
  List.iter
    (fun (u, bytes) ->
      match Ops.Tbl.find_opt numbers u with
      | None -> ()
      | Some i ->
          Option.iter
            (fun b ->
              Rig.Buffer.blit_from_string bytes 0 b 0 (String.length bytes))
            p.storage.(i))
    bound

(* The program of [sink], compiled once unless [cores] is given. *)
let program ?cores numbers sink bound stores =
  match cores with
  | Some _ -> link ?cores numbers sink bound stores
  | None -> (
      let key = program_key numbers sink in
      match Hashtbl.find_opt programs key with
      | Some p ->
          refill p numbers bound;
          p
      | None ->
          let p = link numbers sink bound stores in
          Hashtbl.add programs key p;
          p)

let contents b =
  let len = Rig.Buffer.length b in
  let bytes = Bytes.create len in
  Rig.Buffer.blit_to_bytes b 0 bytes 0 len;
  bytes

(* The bytes of each result [(u, tdt, n)] of [outs], the graph [u] of [n]
   elements of [tdt], computed on the host over the storage [bound]: empty where
   [n] is [0]. Given [cores], the graphs are compiled anew. *)
let execute ?cores bound outs =
  let stores =
    List.filter_map
      (fun (u, tdt, n) ->
        if n = 0 then None else Some (Ops.new_buffer cpu n tdt, u, tdt, n))
      outs
  in
  let read =
    if stores = [] then ref []
    else
      let sink = sink stores in
      let numbers = numbered sink in
      let p = program ?cores numbers sink bound stores in
      Tolk_engine.run ~vars:p.vars p.linked [||];
      ref (List.map contents p.results)
  in
  let next (_, _, n) =
    match !read with
    | b :: rest when n > 0 ->
        read := rest;
        b
    | _ -> Bytes.empty
  in
  List.map next outs

let result (type v s) (o : (v, s) out) = (o.node, L.dtype (D.Any o.dt), o.n)

let elements (type v s) (o : (v, s) out) bytes : v array =
  if o.n = 0 then [||] else decode o.dt o.shape bytes

let run (type v s d) ?local ?cores (f : unit -> (v, s, d) Nx.t) : v array =
  let bound, o = lowered ?local (one f) in
  match execute ?cores bound [ result o ] with
  | [ b ] -> elements o b
  | _ -> invalid_arg "Lowered.run: one result"

type graph = One : (unit -> ('v, 's, 'd) Nx.t) -> graph

(* The schedule of each graph is {!execute}'s, made alone; their calls are then
   compiled as one schedule's, whose kernels {!Tolk.Codegen.to_programs} makes
   together. *)
let prime ?local graphs =
  let scheduled outs =
    let stores =
      List.filter_map
        (fun (u, tdt, n) ->
          if n = 0 then None else Some (Ops.new_buffer cpu n tdt, u, tdt, n))
        outs
    in
    if stores = [] then None else Some (fst (schedule (sink stores)))
  in
  let linear (One f) =
    let _, o = lowered ?local (one f) in
    scheduled [ result o ]
  in
  match List.filter_map linear graphs with
  | [] -> ()
  | first :: _ as linears ->
      let calls = List.concat_map Ops.src linears in
      let targets _ = host_target () in
      ignore (Realize.compile_linear ~targets (Ops.replace first ~src:calls))

let primed ?local graphs examples f =
  let made = lazy (prime ?local (List.concat_map graphs examples)) in
  fun c ->
    Lazy.force made;
    f c

let nodes f =
  let _, o = lowered (one f) in
  List.length (Ops.toposort ~calls:Skip o.node)

let map (type s d) (f : Ops.t -> Ops.t) (x : (float, s, d) Nx.t) : float array =
  let bound = ref [] in
  let dt = Nx.dtype x and shape = Nx.shape x in
  let node = f (bind ~direct:true ~device:cpu bound x) in
  let o = { node; dt; shape; n = numel shape } in
  match execute !bound [ result o ] with
  | [ b ] -> elements o b
  | _ -> invalid_arg "Lowered.map: one result"

(* The calls among [us], each launch's members in its place. *)
let calls us =
  List.concat_map
    (fun c ->
      if not (Op.equal (Ops.op c) Op.Call) then []
      else if Shape.is_launch (Ops.body c) then Shape.launch_members c
      else [ c ])
    us

(* The estimates of [call]: its kernel's for [renderer], made up to its
   instructions as its program would be, or a copy's bytes. *)
let estimates renderer call =
  let ast = Ops.body call in
  match (Ops.op ast, Ops.arg ast) with
  | Op.Sink, Kernel _ -> (
      match Ops.arg (Ops.nth (Codegen.linearize ast renderer) 0) with
      | Kernel { estimates = Some e; _ } -> e
      | _ -> Renderer.Estimates.zero)
  | _ -> Realize.estimate_uop call

let work (type v s d) ?target (f : unit -> (v, s, d) Nx.t) =
  let place =
    match target with
    | None -> on_host
    | Some t ->
        { on_host with renderer = Lazy.from_val (Cstyle.without_compiler t) }
  in
  let renderer = Lazy.force place.renderer in
  let _, o = lowered ~direct:true ~place (one f) in
  let cores = [ (cpu, Heuristic.cores renderer) ] in
  let linear, vars = schedule ~cores (stored o) in
  if not (List.for_all (fun c -> Op.equal (Ops.op c) Op.Call) (Ops.src linear))
  then invalid_arg "Lowered.work: a schedule with loops";
  let count x = Shape.sym_infer x vars in
  List.fold_left
    (fun (ops, lds) call ->
      let e = estimates renderer call in
      (ops + count e.ops, lds + count e.lds))
    (0, 0)
    (calls (Ops.src linear))

let kernels (type v s d) ?cores (f : unit -> (v, s, d) Nx.t) =
  let _, o = lowered ~direct:true (one f) in
  List.length (calls (Ops.src (fst (schedule ?cores (stored o)))))

(* The sink of each kernel of the schedule of the stores [sink] makes on
   [device], whose tensor cores are [renderer]'s, in the order the schedule
   calls them. *)
let scheduled ~device renderer sink =
  let linear, _ =
    Schedule.create_linear_with_vars
      ~cores:[ (device, Heuristic.cores renderer) ]
      sink
  in
  let kernel c =
    let ast = Ops.body c in
    match (Ops.op ast, Ops.arg ast) with
    | Op.Sink, Kernel _ -> Some ast
    | _ -> None
  in
  List.filter_map kernel (calls (Ops.toposort ~calls:Skip linear))

(* Each kernel of {!scheduled}, rendered by [renderer]: its name, sink and
   source. *)
let rendered ~device renderer sink =
  let render ast =
    let p = Codegen.linearize ast renderer in
    let name = match Ops.arg (Ops.nth p 0) with Kernel k -> k.name | _ -> "" in
    (name, ast, renderer.render (Ops.src (Ops.nth p 1)))
  in
  List.map render (scheduled ~device renderer sink)

(* The name and sink of each kernel of its own that the graph [u] calls
   ({!Shape.custom_kernel}), in {!Ops.toposort}'s order of the calls. *)
let own_kernels_of u =
  List.filter_map
    (fun c ->
      if not (Op.equal (Ops.op c) Op.Call) then None
      else
        let ast = Ops.body c in
        match (Ops.op ast, Ops.arg ast) with
        | Op.Sink, Kernel k -> Some (k.name, ast)
        | _ -> None)
    (Ops.toposort ~calls:Skip u)

(* [target]'s renderer, made without its toolchain, and the place of a
   graph lowered for its device. *)
let off_host (target : Helpers.Target.t) =
  let renderer = Cstyle.without_compiler target in
  (renderer, { device = target.device; renderer = Lazy.from_val renderer; host = false })

let sources (type v s d) ~(target : Helpers.Target.t)
    (f : unit -> (v, s, d) Nx.t) =
  let renderer, place = off_host target in
  let _, o = lowered ~direct:true ~local:false ~place (one f) in
  rendered ~device:place.device renderer (stored ~device:place.device o)

let own_kernels (type v s d) ~(target : Helpers.Target.t)
    (f : unit -> (v, s, d) Nx.t) =
  let _, place = off_host target in
  let _, o = lowered ~direct:true ~local:false ~place (one f) in
  own_kernels_of o.node

let derivative_kernels (type v s d) ~(target : Helpers.Target.t)
    (f : unit -> (v, s, d) Nx.t) =
  let _, place = off_host target in
  let found = ref [] in
  let _ = lowered ~direct:true ~local:false ~place ~transposed:found (one f) in
  List.map
    (Option.map (fun us -> own_kernels_of (Ops.sink (Array.to_list us))))
    !found
