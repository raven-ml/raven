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
  | Svd
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

(* The results of [op] over the operands' nodes [xs], one per destination node
   of [dsts], of its dtype and shape. *)
let lower op xs dsts =
  match (op, xs) with
  | Cast, [ x ] -> [ Lower_arith.cast (Ops.dtype (List.hd dsts)) x ]
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
  | Svd, [ x ] ->
      (* The factors are full when [u] and [vt] are square. *)
      let square f =
        match List.rev (Ops.max_shape f) with
        | n :: m :: _ -> n = m
        | _ -> false
      in
      let full_matrices =
        List.for_all square [ List.hd dsts; List.nth dsts 2 ]
      in
      let u, s, vt = Lower_linalg.svd ~full_matrices x in
      [ u; s; vt ]
  | Solve_triangular (upper, transpose, unit_diag), [ a; b ] ->
      [ Lower_linalg.solve_triangular ~upper ~transpose ~unit_diag a b ]
  | _ -> invalid_arg "an operation of other operands"

(* Layouts *)

type operand = A : ('a, 'b) Nx_array.t -> operand

(* How a program reads an array's elements: their dtype, the view over the run
   of its storage the program binds ({!Lower.within}), and where that run starts
   within 16 bytes of memory ({!Lower.phase}), which the program's vector
   accesses are aligned to. *)
type layout = { dtype : Dtype.t; view : View.t; phase : int }

let tolk_dtype what (A a) =
  match Lower.dtype a.dtype with
  | Some dt -> dt
  | None -> refuse what "no %s" (Nx_dtype.to_string a.dtype)

let layout what (A a as x) =
  let dtype = tolk_dtype what x and v = a.view in
  if View.numel v = 0 then
    { dtype; view = View.create (View.shape v); phase = 0 }
  else
    let start, _ = Lower.span dtype v in
    {
      dtype;
      view = Lower.within dtype v;
      phase = Lower.phase dtype a.buffer start;
    }

type key = {
  op : op;
  inputs : layout list;
  outputs : layout list;
  target : Helpers.Target.t;
}

(* Devices *)

let lock = Mutex.create ()

(* What a device's programs are compiled for, and the dtypes they compute. *)
type target = { target : Helpers.Target.t; dtypes : Dtype.t list }

let runs_on d =
  d != Nx_device.disk
  &&
  match Engine.target d with
  | exception Invalid_argument _ -> false
  | t ->
      t.device = "CPU"
      || Option.is_some
           (Engine.device [ ("DEVICE", d) ] "DEVICE").compiler.queues

(* [memo cell latch k make] is [k]'s value in [cell], made by [make] the first
   time, under [latch]. [cell] is read without [latch], which only guards its
   additions. *)
let memo cell latch k make =
  match List.assq_opt k (Atomic.get cell) with
  | Some v -> v
  | None -> (
      Mutex.protect latch @@ fun () ->
      match List.assq_opt k (Atomic.get cell) with
      | Some v -> v
      | None ->
          let v = make () in
          Atomic.set cell ((k, v) :: Atomic.get cell);
          v)

let targets = Atomic.make []

(* [target what d] is [d]'s target, for the kernel [what], which refuses a
   device the backend does not run on. *)
let target what d =
  memo targets lock d @@ fun () ->
  if not (runs_on d) then
    refuse what "%s runs no compiled program" (Nx_device.name d);
  let target = Engine.target d in
  let r =
    match Device.renderer ~arch:target.arch target.device with
    | Ok r -> r
    | Error why -> failwith why
  in
  { target; dtypes = Renderer.supported_dtypes r }

(* Programs *)

(* A compiled program, in which its device has the name [name], the storage of
   the [i]th of its operands and destinations is the parameter of slot
   [slots.(i)] ([-1] for an empty array), and its links on each device it ran
   on, taken in turn. *)
type program = {
  linear : Ops.t;
  name : string;
  slots : int array;
  nslots : int;
  latch : Mutex.t;
  links : (Nx_device.t * (Engine.t array * int Atomic.t)) list Atomic.t;
}

(* Runs of one link are serialized: a program keeps this many on each device, so
   that as many runs of it can be queued at once. On an M1 Max, a chain of one
   operation takes 240 us a run with one link, 53 with 4, 25 with 16 and 24 with
   64: then Metal's submission of each command buffer bounds it. *)
let links = 16

(* [compile key d] is the program of [key], compiled through [d], a device of
   [key]'s target and host. It names its device after the target: a link binds
   the name to the device it runs on, and the engine names its host. *)
let compile (key : key) d =
  let name = key.target.device in
  let device = Ops.Single name in
  (* Each layout's node, and its run's buffer if it has elements. *)
  let node (l : layout) =
    if View.numel l.view = 0 then
      ( Lower.broadcast
          (Ops.const ~dtype:l.dtype (`Int Z.zero))
          (View.shape l.view),
        None )
    else
      let start, span = Lower.span l.dtype l.view in
      let b = Ops.new_buffer ~phase:l.phase device span l.dtype in
      (Lower.strided b l.view start, Some b)
  in
  let operands = List.map node key.inputs
  and outs = List.map node key.outputs in
  let results = lower key.op (List.map fst operands) (List.map fst outs) in
  let stores =
    List.map2
      (fun (view, _) value ->
        if Ops.max_shape value <> Ops.max_shape view then
          invalid_arg "a lowered result of another shape than its destination";
        Ops.after view [ Ops.store view value ])
      outs results
  in
  let linear, _ = Schedule.create_linear_with_vars (Ops.sink stores) in
  let buffers = List.filter_map snd (operands @ outs) in
  let devices = Engine.device [ (name, d) ] in
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
    slots = Array.of_list (List.map (fun (_, b) -> slot b) (operands @ outs));
    nslots = List.length buffers;
    latch = Mutex.create ();
    links = Atomic.make [];
  }

(* [p]'s next link on [d], linked there the first time. *)
let link p d =
  let links, next =
    memo p.links p.latch d @@ fun () ->
    let devices = Engine.device [ (p.name, d) ] in
    (Array.init links (fun _ -> Engine.link ~devices p.linear), Atomic.make 0)
  in
  links.(Atomic.fetch_and_add next 1 mod Array.length links)

(* By the name of the host of the device a program compiled through, which its
   host programs name, and its key. Hashed through the whole key: [Hashtbl.hash]
   stops before most of its shapes, and keys that differ only there would share
   a bucket. *)
module Programs = Memo.Make (struct
  type t = string * key

  let equal = ( = )
  let hash = Hashtbl.hash_param 256 512
end)

let programs : program Programs.t = Programs.create ()

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
  Programs.find programs
    (Nx_device.name (Nx_device.host_of d), key)
    ~miss:(fun () -> check what t d (arrays @ dsts) (key.inputs @ key.outputs))
    (fun () ->
      Nx_device.Profile.span ("compile " ^ what) (fun () -> compile key d))

(* The buffer of the run of [a]'s storage its parameter binds. *)
let slice (A a) = Lower.run (Option.get (Lower.dtype a.dtype)) a.view a.buffer

(* [run what op arrays dsts] computes [op] over [arrays] into [dsts], on the
   device of [dsts]. *)
let run what op arrays dsts =
  let d =
    match dsts with A a :: _ -> B.device a.buffer | [] -> assert false
  in
  let t = target what d and all = arrays @ dsts in
  let inputs = List.map (layout what) arrays
  and outputs = List.map (layout what) dsts in
  if List.for_all (fun (A a) -> View.numel a.view = 0) dsts then
    check what t d all (inputs @ outputs)
  else
    let key = { op; inputs; outputs; target = t.target } in
    let p = program what key t d arrays dsts in
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
  let svd x ~u ~s ~vt = run "svd" Svd [ A x ] [ A u; A s; A vt ]
  let eig _ ~values:_ ~vectors:_ = refused "eig"
  let eigh _ ~values:_ ~vectors:_ = refused "eigh"

  let solve_triangular ~upper ~transpose ~unit_diag a b ~dst =
    run "solve_triangular"
      (Solve_triangular (upper, transpose, unit_diag))
      [ A a; A b ] [ A dst ]
end

let backend = Nx_backend.make (module Kernels)
