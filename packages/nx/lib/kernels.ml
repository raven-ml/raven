(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array
open Value

(* Kernels

   nx allocates each result, C-contiguous from its first element, and a
   backend's kernel writes it, reading its operands' arrays on one device. *)

(* Where kernels run: how a result is allocated there, and each operand's array
   there. *)
type env = {
  device : Device.t;
  kernels : (module Nx_backend.S);
  alloc : 'a 'b. ('a, 'b) Nx_dtype.t -> int array -> ('a, 'b) Nx_array.t;
  arr : 'a 'b. ('a, 'b) t -> ('a, 'b) Nx_array.t;
}

let host =
  {
    device = Device.host;
    kernels = Nx_backend.kernels Nx_cpu.backend;
    alloc;
    arr = Place.host_of;
  }

let shape_of (a : ('a, 'b) Nx_array.t) = View.shape a.view

(* Kernels read their operands' memory under read claims, so that no compiled
   call lends it to a result meanwhile. Claims are released however the kernel
   ends, and none allocates. *)

let claim (a : ('a, 'b) Nx_array.t) = Nx_device.Buffer.Claim.read a.buffer
let release (a : ('a, 'b) Nx_array.t) = Nx_device.Buffer.Claim.release a.buffer

let release2 a b =
  release a;
  release b

let release3 a b c =
  release2 a b;
  release c

let claim2 a b =
  claim a;
  match claim b with
  | () -> ()
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt

let claim3 a b c =
  claim2 a b;
  match claim c with
  | () -> ()
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt

let rec claim_all = function
  | [] -> ()
  | a :: rest -> (
      claim a;
      match claim_all rest with
      | () -> ()
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          release a;
          Printexc.raise_with_backtrace e bt)

let release_all xs = List.iter release xs

(* Each operation's results, allocated, and written by [e]'s kernels. *)

let unary (e : env) k a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.unary k a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let binary (e : env) k a b =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim2 a b;
  (match K.binary k a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

let compare (e : env) k a b =
  let (module K) = e.kernels in
  let dst = e.alloc Nx_dtype.Bool (shape_of a) in
  claim2 a b;
  (match K.compare k a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

let where (e : env) c a b =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim3 c a b;
  (match K.where c a b ~dst with
  | () -> release3 c a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release3 c a b;
      Printexc.raise_with_backtrace e bt);
  dst

let fma (e : env) a b c =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim3 a b c;
  (match K.fma a b c ~dst with
  | () -> release3 a b c
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release3 a b c;
      Printexc.raise_with_backtrace e bt);
  dst

let reduce (e : env) k axes a =
  let (module K) = e.kernels in
  let dst =
    e.alloc a.dtype (Shape.reduce_output_shape (shape_of a) axes false)
  in
  claim a;
  (match K.reduce k ~axes a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let scan (e : env) k axis a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.scan k ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let arg_reduce (e : env) k axis a =
  let (module K) = e.kernels in
  let dst =
    e.alloc Nx_dtype.Int64
      (Shape.reduce_output_shape (shape_of a) [| axis |] false)
  in
  claim a;
  (match K.arg_reduce k ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let sort (e : env) descending axis a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.sort ~descending ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let argsort (e : env) descending axis a =
  let (module K) = e.kernels in
  let dst = e.alloc Nx_dtype.Int64 (shape_of a) in
  claim a;
  (match K.argsort ~descending ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let group (e : env) a =
  let (module K) = e.kernels in
  let dst = e.alloc Nx_dtype.Int64 [| (shape_of a).(0) |] in
  claim a;
  (match K.group a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let pad (e : env) padding v a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (Op.pad_shape padding (shape_of a)) in
  claim a;
  (match K.pad padding v a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let cat (e : env) axis xs =
  let (module K) = e.kernels in
  let shape = Op.cat_shape axis (List.map shape_of xs) in
  let dst = e.alloc (List.hd xs).dtype shape in
  claim_all xs;
  (match K.cat ~axis xs ~dst with
  | () -> release_all xs
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release_all xs;
      Printexc.raise_with_backtrace e bt);
  dst

let cast (e : env) dtype a =
  let (module K) = e.kernels in
  let dst = e.alloc dtype (shape_of a) in
  claim a;
  (match K.cast a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let threefry (e : env) key ctr =
  let (module K) = e.kernels in
  let dst = e.alloc Nx_dtype.Int32 (shape_of ctr) in
  claim2 key ctr;
  (match K.threefry key ctr ~dst with
  | () -> release2 key ctr
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 key ctr;
      Printexc.raise_with_backtrace e bt);
  dst

let gather (e : env) axis indices data =
  let (module K) = e.kernels in
  let dst = e.alloc data.dtype (shape_of indices) in
  claim2 indices data;
  (match K.gather ~axis indices data ~dst with
  | () -> release2 indices data
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 indices data;
      Printexc.raise_with_backtrace e bt);
  dst

let scatter (e : env) mode unique axis indices updates into =
  let (module K) = e.kernels in
  let dst = e.alloc into.dtype (shape_of into) in
  claim3 indices updates into;
  (match K.scatter ~mode ~unique ~axis ~indices ~updates into ~dst with
  | () -> release3 indices updates into
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release3 indices updates into;
      Printexc.raise_with_backtrace e bt);
  dst

let update (e : env) a starts v =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim3 a starts v;
  (match K.update a ~starts v ~dst with
  | () -> release3 a starts v
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release3 a starts v;
      Printexc.raise_with_backtrace e bt);
  dst

let unfold (e : env) kernel_size stride dilation padding a =
  let (module K) = e.kernels in
  let dst =
    e.alloc a.dtype
      (Op.unfold_shape kernel_size stride dilation padding (shape_of a))
  in
  claim a;
  (match K.unfold ~kernel_size ~stride ~dilation ~padding a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let fold (e : env) output_size kernel_size stride dilation padding a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (Op.fold_shape output_size (shape_of a)) in
  claim a;
  (match K.fold ~output_size ~kernel_size ~stride ~dilation ~padding a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let matmul (e : env) a b =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (Op.matmul_shape (shape_of a) (shape_of b)) in
  claim2 a b;
  (match K.matmul a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

let fft (e : env) inverse axes a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.fft ~inverse ~axes a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let rfft (e : env) dtype axes a =
  let (module K) = e.kernels in
  let dst = e.alloc dtype (Op.rfft_shape axes (shape_of a)) in
  claim a;
  (match K.rfft ~axes a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let irfft (e : env) dtype axes s a =
  let (module K) = e.kernels in
  let dst = e.alloc dtype (Op.irfft_shape axes s (shape_of a)) in
  claim a;
  (match K.irfft ~axes ~s a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let contiguous (e : env) a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.contiguous a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

(* [bitcast e dtype a] is [a]'s bits read as elements of [dtype]: at [a]'s
   width, the same view; [k] times narrower, each element as [k] along a new
   last axis; [k] times wider, each run of [k] along the last axis, of [k], as
   one element: a view of [a]'s memory when [a] is C-contiguous from a first
   element whose bits start at a multiple of [dtype]'s width in memory, and of a
   C-contiguous copy of [a] that [e] makes otherwise. Widths count bits, so
   [bit] is an eighth of [uint8]. *)
let bitcast (type a b c d) (e : env) (dtype : (c, d) Nx_dtype.t)
    (a : (a, b) Nx_array.t) : (c, d) Nx_array.t =
  let bits dt = Nx_dtype.Scalar.(bitsize (of_dtype dt)) in
  let w = bits a.dtype and w' = bits dtype in
  let scalar = Nx_dtype.Scalar.of_dtype dtype in
  let over (a : (a, b) Nx_array.t) view : (c, d) Nx_array.t =
    let n = Nx_device.Buffer.nbytes a.buffer * 8 / w' in
    { dtype; view; buffer = Nx_device.Buffer.view a.buffer ~offset:0 scalar n }
  in
  let v = a.view in
  if w' = w then over a v
  else if w' < w then
    let k = w / w' in
    over a
      (View.create
         ~offset:(View.offset v * k)
         ~strides:(Array.append (Array.map (( * ) k) (View.strides v)) [| 1 |])
         (Array.append (View.shape v) [| k |]))
  else
    let k = w' / w in
    (* The first element's bits, counted from a byte of the buffer's memory
       aligned to [dtype]'s width when it is a whole number of bytes. *)
    let first = View.offset v * w in
    let aligned =
      let align = Int.max 1 (w' / 8) in
      let byte =
        Nativeint.(add (Nx_device.Buffer.address a.buffer) (of_int (first / 8)))
      in
      first mod Int.min 8 w' = 0 && Nativeint.(rem byte (of_int align)) = 0n
    in
    let a, first =
      if View.is_c_contiguous v && aligned then (a, first)
      else (contiguous e a, 0)
    in
    let s = View.shape a.view in
    let skip = first mod 8 / w' in
    {
      dtype;
      view = View.create ~offset:skip (Array.sub s 0 (Array.length s - 1));
      buffer =
        Nx_device.Buffer.view a.buffer ~offset:(first / 8) scalar
          (skip + (View.numel a.view / k));
    }

let cholesky (e : env) upper a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.cholesky ~upper a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

(* The batch axes of a matrix of shape [..., m, n], and [m] and [n]. *)
let matrix a =
  let s = shape_of a in
  let r = Array.length s in
  (Array.sub s 0 (r - 2), s.(r - 2), s.(r - 1))

let qr (e : env) reduced a =
  let (module K) = e.kernels in
  let batch, m, n = matrix a in
  let k = Int.min m n in
  let dims r c = Array.append batch [| r; c |] in
  let q = e.alloc a.dtype (if reduced then dims m k else dims m m) in
  let r = e.alloc a.dtype (if reduced then dims k n else dims m n) in
  claim a;
  (match K.qr ~reduced a ~q ~r with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (q, r)

let lu (e : env) a =
  let (module K) = e.kernels in
  let batch, m, n = matrix a in
  let lu = e.alloc a.dtype (shape_of a) in
  let pivots = e.alloc Nx_dtype.Int64 (Array.append batch [| Int.min m n |]) in
  let perm = e.alloc Nx_dtype.Int64 (Array.append batch [| m |]) in
  claim a;
  (match K.lu a ~lu ~pivots ~perm with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (lu, pivots, perm)

let svd (e : env) full_matrices a =
  let (module K) = e.kernels in
  let batch, m, n = matrix a in
  let k = Int.min m n in
  let dims r c = Array.append batch [| r; c |] in
  let u = e.alloc a.dtype (if full_matrices then dims m m else dims m k) in
  let s = e.alloc Nx_dtype.Float64 (Array.append batch [| k |]) in
  let vt = e.alloc a.dtype (if full_matrices then dims n n else dims k n) in
  claim a;
  (match K.svd a ~u ~s ~vt with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (u, s, vt)

let eig (e : env) vectors a =
  let (module K) = e.kernels in
  let batch, _, n = matrix a in
  let values = e.alloc Nx_dtype.Complex128 (Array.append batch [| n |]) in
  let vectors =
    if vectors then Some (e.alloc Nx_dtype.Complex128 (shape_of a)) else None
  in
  claim a;
  (match K.eig a ~values ~vectors with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (values, vectors)

let eigh (e : env) vectors a =
  let (module K) = e.kernels in
  let batch, _, n = matrix a in
  let values = e.alloc Nx_dtype.Float64 (Array.append batch [| n |]) in
  let vectors = if vectors then Some (e.alloc a.dtype (shape_of a)) else None in
  claim a;
  (match K.eigh a ~values ~vectors with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (values, vectors)

let solve_triangular (e : env) upper transpose unit_diag a b =
  let (module K) = e.kernels in
  let dst = e.alloc b.dtype (shape_of b) in
  claim2 a b;
  (match K.solve_triangular ~upper ~transpose ~unit_diag a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

(* [x]'s array on [d]: its storage there, through its view. *)
let local (type a b) d (x : (a, b) t) : (a, b) Nx_array.t =
  match x with
  | Placed r -> (
      match Cell.state r.r_cell with
      | Live bufs ->
          {
            dtype = r.r_dtype;
            view = r.r_view;
            buffer = buffer_on r.r_cell bufs d;
          }
      | Consumed k -> consumed k)
  | Host _ -> invalid_arg "Nx: a host value has no device array"
  | Traced _ -> outside_trace ()

(* Computing on [d] with [kernels]: results in [d]'s memory. *)
let on kernels d =
  let alloc (type a b) (dtype : (a, b) Nx_dtype.t) shape : (a, b) Nx_array.t =
    let n = Array.fold_left ( * ) 1 shape in
    let s = Nx_dtype.Scalar.of_dtype dtype in
    {
      dtype;
      view = View.create shape;
      buffer = Nx_device.Buffer.create (Device.memory d) s n;
    }
  in
  { device = d; kernels; alloc; arr = (fun x -> local d x) }
