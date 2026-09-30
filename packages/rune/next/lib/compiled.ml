(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_next
module B = Nx_device.Buffer
module View = Nx_array.View
module Engine = Tolk_next_engine

let name = "compiled"

let refuse what fmt =
  Printf.ksprintf
    (fun s -> raise (Nx_backend.Refused (name ^ ": " ^ what ^ ": " ^ s)))
    fmt

(* Operations *)

(* An operation with its static arguments: what, beside its operands' layouts,
   its program depends on. *)
type op =
  | Unary of Nx_backend.unary
  | Binary of Nx_backend.binary
  | Compare of Nx_backend.compare
  | Where
  | Cast
  | Threefry
  | Reduce of Nx_backend.reduce * int list
  | Scan of Nx_backend.reduce * int
  | Arg_reduce of Nx_backend.arg_reduce * int
  | Sort of bool * int
  | Argsort of bool * int
  | Pad of (int * int) array * fill
  | Cat of int
  | Contiguous
  | Gather of int
  | Scatter of [ `Set | `Add ] * bool * int
  | Update
  | Unfold of int array * int array * int array * (int * int) array
  | Fold of int array * int array * int array * int array * (int * int) array
  | Matmul
  | Cholesky of bool
  | Qr of bool
  | Lu
  | Svd of bool
  | Solve_triangular of bool * bool * bool

(* A fill value as a key holds it: a float by its bits, since [-0.] and each NaN
   pad as themselves, and structural equality would take them for [0.] and for
   each other. *)
and fill = Float_bits of int64 | Value of Dtype.const

let fill = function
  | `Float f -> Float_bits (Int64.bits_of_float f)
  | c -> Value c

let fill_const = function
  | Float_bits b -> `Float (Int64.float_of_bits b)
  | Value c -> c

(* The results of [op] over the operands' nodes [xs], one per destination of the
   dtypes [dsts]. *)
let lower op xs dsts =
  match (op, xs) with
  | Cast, [ x ] -> [ Lower_arith.cast (List.hd dsts) x ]
  | Unary k, [ x ] -> [ Lower_arith.unary k x ]
  | Binary k, [ x; y ] -> [ Lower_arith.binary k x y ]
  | Compare k, [ x; y ] -> [ Lower_arith.compare k x y ]
  | Where, [ c; x; y ] -> [ Ops.where c x y ]
  | Threefry, [ key; counter ] -> [ Lower_arith.threefry key counter ]
  | Reduce (k, axes), [ x ] -> [ Lower_reduce.reduce k ~axes x ]
  | Scan (k, axis), [ x ] -> [ Lower_reduce.scan k ~axis x ]
  | Arg_reduce (k, axis), [ x ] -> [ Lower_reduce.arg_reduce k ~axis x ]
  | Sort (descending, axis), [ x ] -> [ Lower_reduce.sort ~descending ~axis x ]
  | Argsort (descending, axis), [ x ] ->
      [ Lower_reduce.argsort ~descending ~axis x ]
  | Pad (padding, v), [ x ] -> [ Lower_index.pad padding (fill_const v) x ]
  | Cat axis, x :: xs -> [ Lower_index.cat axis x xs ]
  | Contiguous, [ x ] -> [ x ]
  | Gather axis, [ indices; x ] -> [ Lower_index.gather axis indices x ]
  | Scatter (mode, unique, axis), [ indices; updates; x ] ->
      [ Lower_index.scatter ~mode ~unique ~axis ~indices ~updates x ]
  | Update, [ x; starts; v ] -> [ Lower_index.update x ~starts v ]
  | Unfold (kernel_size, stride, dilation, padding), [ x ] ->
      [ Lower_index.unfold ~kernel_size ~stride ~dilation ~padding x ]
  | Fold (output_size, kernel_size, stride, dilation, padding), [ x ] ->
      [
        Lower_index.fold ~output_size ~kernel_size ~stride ~dilation ~padding x;
      ]
  | Matmul, [ a; b ] -> [ Lower_linalg.matmul a b ]
  | Cholesky upper, [ x ] -> [ Lower_linalg.cholesky ~upper x ]
  | Qr reduced, [ x ] ->
      let q, r = Lower_linalg.qr ~reduced x in
      [ q; r ]
  | Lu, [ x ] ->
      let lu, pivots, perm = Lower_linalg.lu x in
      [ lu; pivots; perm ]
  | Svd full_matrices, [ x ] ->
      let u, s, vt = Lower_linalg.svd ~full_matrices x in
      [ u; s; vt ]
  | Solve_triangular (upper, transpose, unit_diag), [ a; b ] ->
      [ Lower_linalg.solve_triangular ~upper ~transpose ~unit_diag a b ]
  | _ -> invalid_arg "an operation of other operands"

(* Layouts *)

type operand = A : ('a, 'b) Nx_array.t -> operand

(* How a program reads an array's elements: their dtype, the view's shape and
   strides, and where its run of elements starts modulo the elements of 16 bytes
   ({!Lower.span}). *)
type layout = {
  dtype : Dtype.t;
  shape : int array;
  strides : int array;
  phase : int;
}

let tolk_dtype what (A a) =
  match Lower.dtype a.dtype with
  | Some dt -> dt
  | None -> refuse what "no %s" (Nx_dtype.to_string a.dtype)

let layout what (A a as x) =
  let dtype = tolk_dtype what x and v = a.view in
  let phase =
    if View.numel v = 0 then 0
    else
      let start, _ = Lower.span dtype v in
      fst (View.extent v) - start
  in
  { dtype; shape = View.shape v; strides = View.strides v; phase }

(* A program's key: the layouts are the operands', then the destinations'. *)
type key = { op : op; layouts : layout list; target : Helpers.Target.t }

(* Devices *)

let lock = Mutex.create ()

(* What a device's programs are compiled for, and the dtypes they compute. *)
type target = { target : Helpers.Target.t; dtypes : Dtype.t list }

(* The engine's view of [d], named [name], and of its host, named [host]. *)
let engine_devices ~name ~host d =
  let h = Nx_device.host_of d in
  let named = if h == d then [ (name, d) ] else [ (name, d); (host, h) ] in
  Engine.device named

let runs_on d =
  d != Nx_device.disk
  &&
  match Engine.target d with
  | exception Invalid_argument _ -> false
  | t ->
      t.device = "CPU"
      || Option.is_some
           (engine_devices ~name:"DEVICE" ~host:"HOST" d "DEVICE").compiler
             .queues

(* Read without the lock, which only guards their additions. *)
let targets = Atomic.make []

(* [target what d] is [d]'s target, for the kernel [what], which refuses a
   device the backend does not run on. *)
let target what d =
  match List.assq_opt d (Atomic.get targets) with
  | Some t -> t
  | None -> (
      Mutex.protect lock @@ fun () ->
      match List.assq_opt d (Atomic.get targets) with
      | Some t -> t
      | None ->
          if not (runs_on d) then
            refuse what "%s runs no compiled program" (Nx_device.name d);
          let target = Engine.target d in
          let r =
            match Device.renderer ~arch:target.arch target.device with
            | Ok r -> r
            | Error why -> failwith why
          in
          let t = { target; dtypes = Renderer.supported_dtypes r } in
          Atomic.set targets ((d, t) :: Atomic.get targets);
          t)

(* Programs *)

(* A compiled program, in which its device and its host have the names [name]
   and [host], the storage of the [i]th of its operands and destinations is the
   parameter of slot [slots.(i)] ([-1] for an empty array), and its links on
   each device it ran on, taken in turn. [links] is read without [latch], which
   only guards its additions. *)
type program = {
  linear : Ops.t;
  name : string;
  host : string;
  slots : int array;
  nslots : int;
  latch : Mutex.t;
  links : (Nx_device.t * Engine.t array * int Atomic.t) list Atomic.t;
}

(* Runs of one link are serialized: a program keeps this many on each device, so
   that as many runs of it can be queued at once. On an M1 Max, a chain of one
   operation takes 240 us a run with one link, 53 with 4, 25 with 16 and 24 with
   64: then Metal's submission of each command buffer bounds it. *)
let links = 16

let zeros dt shape =
  let c = Ops.const ~dtype:dt (`Int Z.zero) in
  let shape = Array.to_list shape in
  Ops.expand
    (Ops.reshape c (List.map (fun _ -> Ops.Int 1) shape))
    (List.map (fun n -> Ops.Int n) shape)

let compile key d arrays dsts =
  let operands, outs =
    List.partition_map Fun.id
      (List.mapi
         (fun i l ->
           if i < List.length arrays then Either.Left l else Either.Right l)
         key.layouts)
  in
  let name = Nx_device.name d and host = Nx_device.name (Nx_device.host_of d) in
  let device = Ops.Single name in
  (* Each array's node, and its storage's buffer if it has elements. *)
  let node (A a) (l : layout) =
    if View.numel a.view = 0 then (zeros l.dtype l.shape, None)
    else
      let start, span = Lower.span l.dtype a.view in
      let b = Ops.new_buffer device span l.dtype in
      (Lower.strided b a.view start, Some b)
  in
  let operands = List.map2 node arrays operands
  and results = List.map (fun (l : layout) -> l.dtype) outs
  and outs = List.map2 node dsts outs in
  let results = lower key.op (List.map fst operands) results in
  let stores =
    List.map2
      (fun (view, _) value -> Ops.after view [ Ops.store view value ])
      outs results
  in
  let linear, _ = Schedule.create_linear_with_vars (Ops.sink stores) in
  let buffers = List.filter_map snd (operands @ outs) in
  let devices = engine_devices ~name ~host d in
  let linear =
    Jit.jit_lower
      ~devices:(fun n -> (devices n).compiler)
      ~held_bufs:[] ~inputs:buffers linear
  in
  let slot = function
    | None -> -1
    | Some b -> Option.get (List.find_index (fun b' -> b' == b) buffers)
  in
  {
    linear;
    name;
    host;
    slots = Array.of_list (List.map (fun (_, b) -> slot b) (operands @ outs));
    nslots = List.length buffers;
    latch = Mutex.create ();
    links = Atomic.make [];
  }

(* [p]'s next link on [d], linked there the first time. *)
let rec link p d =
  match List.find_opt (fun (d', _, _) -> d' == d) (Atomic.get p.links) with
  | Some (_, links, next) ->
      links.(Atomic.fetch_and_add next 1 mod Array.length links)
  | None ->
      Mutex.protect p.latch (fun () ->
          if not (List.exists (fun (d', _, _) -> d' == d) (Atomic.get p.links))
          then begin
            let devices = engine_devices ~name:p.name ~host:p.host d in
            let l = Array.init links (fun _ -> Engine.link ~devices p.linear) in
            Atomic.set p.links ((d, l, Atomic.make 0) :: Atomic.get p.links)
          end);
      link p d

(* The compiled programs, by key, each compiled once under its own latch, so
   that keys compile concurrently. *)
type entry = { latch : Mutex.t; compiled : program option Atomic.t }

let programs : (key, entry) Hashtbl.t = Hashtbl.create 64

(* [check what t d arrays layouts] refuses a dtype that [t], the target of [d],
   does not compute. *)
let check what t d arrays layouts =
  List.iter2
    (fun (A a) (l : layout) ->
      if not (List.exists (Dtype.equal l.dtype) t.dtypes) then
        refuse what "no %s on %s"
          (Nx_dtype.to_string a.dtype)
          (Nx_device.name d))
    arrays layouts

(* The program of [key], compiled the first time, once its dtypes are checked: a
   program that exists passed the check. *)
let program what key t d arrays dsts =
  let e =
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt programs key with
    | Some e -> e
    | None ->
        check what t d (arrays @ dsts) key.layouts;
        let e = { latch = Mutex.create (); compiled = Atomic.make None } in
        Hashtbl.add programs key e;
        e
  in
  match Atomic.get e.compiled with
  | Some p -> p
  | None -> (
      Mutex.protect e.latch @@ fun () ->
      match Atomic.get e.compiled with
      | Some p -> p
      | None ->
          let p =
            Nx_device.Profile.span ("compile " ^ what) (fun () ->
                compile key d arrays dsts)
          in
          Atomic.set e.compiled (Some p);
          p)

(* The buffer of the run of [a]'s storage its parameter binds. *)
let slice (A a) =
  let dt = Option.get (Lower.dtype a.dtype) in
  let start, span = Lower.span dt a.view in
  B.view a.buffer
    ~offset:(start * Dtype.itemsize dt)
    (Nx_dtype.Scalar.of_dtype a.dtype)
    span

(* [run what op arrays dsts] computes [op] over [arrays] into [dsts], on the
   device of [dsts]. *)
let run what op arrays dsts =
  let d =
    match dsts with A a :: _ -> B.device a.buffer | [] -> assert false
  in
  let t = target what d and all = arrays @ dsts in
  let layouts = List.map (layout what) all in
  if List.for_all (fun (A a) -> View.numel a.view = 0) dsts then
    check what t d all layouts
  else
    let p = program what { op; layouts; target = t.target } t d arrays dsts in
    let slots = Array.make p.nslots [] in
    List.iteri
      (fun i x ->
        let slot = p.slots.(i) in
        if slot >= 0 then slots.(slot) <- [ slice x ])
      all;
    Engine.run (link p d) slots

(* Kernels *)

let refused what = refuse what "not compiled"

module Kernels = struct
  let name = name
  let runs_on = runs_on
  let unary k x ~dst = run "unary" (Unary k) [ A x ] [ A dst ]
  let binary k a b ~dst = run "binary" (Binary k) [ A a; A b ] [ A dst ]
  let compare k a b ~dst = run "compare" (Compare k) [ A a; A b ] [ A dst ]
  let where c a b ~dst = run "where" Where [ A c; A a; A b ] [ A dst ]
  let cast x ~dst = run "cast" Cast [ A x ] [ A dst ]

  let threefry key counter ~dst =
    run "threefry" Threefry [ A key; A counter ] [ A dst ]

  let reduce k ~axes x ~dst =
    run "reduce" (Reduce (k, Array.to_list axes)) [ A x ] [ A dst ]

  let scan k ~axis x ~dst = run "scan" (Scan (k, axis)) [ A x ] [ A dst ]

  let arg_reduce k ~axis x ~dst =
    run "arg_reduce" (Arg_reduce (k, axis)) [ A x ] [ A dst ]

  let sort ~descending ~axis x ~dst =
    run "sort" (Sort (descending, axis)) [ A x ] [ A dst ]

  let argsort ~descending ~axis x ~dst =
    run "argsort" (Argsort (descending, axis)) [ A x ] [ A dst ]

  let pad padding v (x : ('a, 'b) Nx_array.t) ~dst =
    run "pad" (Pad (padding, fill (Lower.const x.dtype v))) [ A x ] [ A dst ]

  let cat ~axis xs ~dst =
    run "cat" (Cat axis) (List.map (fun x -> A x) xs) [ A dst ]

  let contiguous x ~dst = run "contiguous" Contiguous [ A x ] [ A dst ]

  let gather ~axis indices x ~dst =
    run "gather" (Gather axis) [ A indices; A x ] [ A dst ]

  let scatter ~mode ~unique ~axis ~indices ~updates x ~dst =
    run "scatter"
      (Scatter (mode, unique, axis))
      [ A indices; A updates; A x ]
      [ A dst ]

  let update x ~starts v ~dst =
    run "update" Update [ A x; A starts; A v ] [ A dst ]

  let unfold ~kernel_size ~stride ~dilation ~padding x ~dst =
    run "unfold"
      (Unfold (kernel_size, stride, dilation, padding))
      [ A x ] [ A dst ]

  let fold ~output_size ~kernel_size ~stride ~dilation ~padding x ~dst =
    run "fold"
      (Fold (output_size, kernel_size, stride, dilation, padding))
      [ A x ] [ A dst ]

  let matmul a b ~dst = run "matmul" Matmul [ A a; A b ] [ A dst ]
  let fft ~inverse:_ ~axes:_ _ ~dst:_ = refused "fft"
  let rfft ~axes:_ _ ~dst:_ = refused "rfft"
  let irfft ~axes:_ ~s:_ _ ~dst:_ = refused "irfft"
  let cholesky ~upper x ~dst = run "cholesky" (Cholesky upper) [ A x ] [ A dst ]
  let qr ~reduced x ~q ~r = run "qr" (Qr reduced) [ A x ] [ A q; A r ]
  let lu x ~lu ~pivots ~perm = run "lu" Lu [ A x ] [ A lu; A pivots; A perm ]

  (* The factors are full when [u] is square and [vt] has [x]'s columns. *)
  let svd (x : ('a, 'b) Nx_array.t) ~(u : ('a, 'b) Nx_array.t) ~s
      ~(vt : ('a, 'b) Nx_array.t) =
    let dim i a = View.dim (View.ndim a.Nx_array.view - i) a.view in
    let full = dim 1 u = dim 2 x && dim 2 vt = dim 1 x in
    run "svd" (Svd full) [ A x ] [ A u; A s; A vt ]

  let eig _ ~values:_ ~vectors:_ = refused "eig"
  let eigh _ ~values:_ ~vectors:_ = refused "eigh"

  let solve_triangular ~upper ~transpose ~unit_diag a b ~dst =
    run "solve_triangular"
      (Solve_triangular (upper, transpose, unit_diag))
      [ A a; A b ] [ A dst ]
end

let backend = Nx_backend.make (module Kernels)
