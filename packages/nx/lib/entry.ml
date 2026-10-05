(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array
open Value

(* Entry functions

   One per constructor of [Op.t]. While no interception is live anywhere, each
   answers directly and builds no operation; otherwise it performs its
   operation. Constants and values over runtime buffers build on them.

   A [bit] operand reaches an operation only where backends compute on bits:
   casts, bitcasts, [And], [Or], [Xor], [Maximum] and [Minimum], and the moves
   ([pad], [cat], [gather], [scatter] with [`Set], [update] and copies). Every
   other operation reads its [bit] operands as [bool] and stores a result of
   their dtype as [bit], so it answers as it does on [bool], raising where it
   raises. *)

let cast dtype x =
  if Intercept.intercepting () then Intercept.perform (Convert (Cast, dtype, x))
  else Dispatch.convert Cast dtype x

let bool x = cast Nx_dtype.Bool x

(* [through_bool f x] is [f] of [x] read as [bool], its result kept as [bit]. *)
let through_bool f x = cast Nx_dtype.Bit (f (bool x))

let unary_op k x =
  if Intercept.intercepting () then Intercept.perform (Unary (k, x))
  else Dispatch.unary k x

let unary (type a b) k (x : (a, b) t) : (a, b) t =
  match dtype x with
  | Nx_dtype.Bit -> through_bool (unary_op k) x
  | _ -> unary_op k x

let binary_op k x y =
  if Intercept.intercepting () then Intercept.perform (Binary (k, x, y))
  else Dispatch.binary k x y

let binary (type a b) (k : Nx_backend.binary) (x : (a, b) t) (y : (a, b) t) :
    (a, b) t =
  match (dtype x, k) with
  | Nx_dtype.Bit, (Add | Sub | Mul | Fdiv | Idiv | Mod | Pow | Atan2) ->
      through_bool (fun x -> binary_op k x (bool y)) x
  | Nx_dtype.Bit, (And | Or | Xor | Maximum | Minimum) -> binary_op k x y
  | _ -> binary_op k x y

let cmp_op k x y =
  if Intercept.intercepting () then Intercept.perform (Compare (k, x, y))
  else Dispatch.compare k x y

let cmp (type a b) k (x : (a, b) t) (y : (a, b) t) =
  match dtype x with
  | Nx_dtype.Bit -> cmp_op k (bool x) (bool y)
  | _ -> cmp_op k x y

let where_op c x y =
  if Intercept.intercepting () then Intercept.perform (Where (c, x, y))
  else Dispatch.where c x y

let where (type a b) c (x : (a, b) t) (y : (a, b) t) : (a, b) t =
  match dtype x with
  | Nx_dtype.Bit -> through_bool (fun x -> where_op c x (bool y)) x
  | _ -> where_op c x y

let fma_op a b c =
  if Intercept.intercepting () then Intercept.perform (Fma (a, b, c))
  else Dispatch.fma a b c

let fma (type a b) (a : (a, b) t) (b : (a, b) t) (c : (a, b) t) : (a, b) t =
  match dtype a with
  | Nx_dtype.Bit -> through_bool (fun a -> fma_op a (bool b) (bool c)) a
  | _ -> fma_op a b c

let reduce_op k axes x =
  if Intercept.intercepting () then Intercept.perform (Reduce (k, axes, x))
  else Dispatch.reduce k axes x

let reduce (type a b) k ~axes (x : (a, b) t) : (a, b) t =
  match dtype x with
  | Nx_dtype.Bit -> through_bool (reduce_op k axes) x
  | _ -> reduce_op k axes x

let scan_op k axis x =
  if Intercept.intercepting () then Intercept.perform (Scan (k, axis, x))
  else Dispatch.scan k axis x

let scan (type a b) k ~axis (x : (a, b) t) : (a, b) t =
  match dtype x with
  | Nx_dtype.Bit -> through_bool (scan_op k axis) x
  | _ -> scan_op k axis x

let arg_reduce_op k axis x =
  if Intercept.intercepting () then Intercept.perform (Arg_reduce (k, axis, x))
  else Dispatch.arg_reduce k axis x

let arg_reduce (type a b) k ~axis (x : (a, b) t) =
  match dtype x with
  | Nx_dtype.Bit -> arg_reduce_op k axis (bool x)
  | _ -> arg_reduce_op k axis x

let sort ~descending ~axis x =
  if Intercept.intercepting () then
    Intercept.perform (Sort { descending; axis; x })
  else Dispatch.sort descending axis x

let argsort ~descending ~axis x =
  if Intercept.intercepting () then
    Intercept.perform (Argsort { descending; axis; x })
  else Dispatch.argsort descending axis x

let group ~by x =
  if Intercept.intercepting () then Intercept.perform (Group { by; x })
  else Dispatch.group by x

let pad padding v x =
  if Intercept.intercepting () then Intercept.perform (Pad (padding, v, x))
  else Dispatch.pad padding v x

let cat ~axis xs =
  if Intercept.intercepting () then Intercept.perform (Cat (axis, xs))
  else Dispatch.cat axis xs

let bitcast dtype x =
  if Intercept.intercepting () then
    Intercept.perform (Convert (Bitcast, dtype, x))
  else Dispatch.convert Bitcast dtype x

let threefry key ctr =
  if Intercept.intercepting () then Intercept.perform (Threefry (key, ctr))
  else Dispatch.threefry key ctr

let gather ~axis indices x =
  if Intercept.intercepting () then
    Intercept.perform (Gather (axis, indices, x))
  else Dispatch.gather axis indices x

let scatter_op mode unique axis indices updates into =
  if Intercept.intercepting () then
    Intercept.perform (Scatter { mode; unique; axis; indices; updates; into })
  else Dispatch.scatter mode unique axis indices updates into

let scatter (type a b) ~mode ~unique ~axis ~indices ~(updates : (a, b) t)
    (into : (a, b) t) : (a, b) t =
  match (dtype into, mode) with
  | Nx_dtype.Bit, (`Add | `Max | `Min) ->
      through_bool (scatter_op mode unique axis indices (bool updates)) into
  | _ -> scatter_op mode unique axis indices updates into

let update x ~starts v =
  if Intercept.intercepting () then Intercept.perform (Update (x, starts, v))
  else Dispatch.update x starts v

let unfold_op kernel_size stride dilation padding x =
  if Intercept.intercepting () then
    Intercept.perform (Unfold { kernel_size; stride; dilation; padding; x })
  else Dispatch.unfold kernel_size stride dilation padding x

let unfold (type a b) ~kernel_size ~stride ~dilation ~padding (x : (a, b) t) :
    (a, b) t =
  match dtype x with
  | Nx_dtype.Bit ->
      through_bool (unfold_op kernel_size stride dilation padding) x
  | _ -> unfold_op kernel_size stride dilation padding x

let fold_op output_size kernel_size stride dilation padding x =
  if Intercept.intercepting () then
    Intercept.perform
      (Fold { output_size; kernel_size; stride; dilation; padding; x })
  else Dispatch.fold output_size kernel_size stride dilation padding x

let fold (type a b) ~output_size ~kernel_size ~stride ~dilation ~padding
    (x : (a, b) t) : (a, b) t =
  match dtype x with
  | Nx_dtype.Bit ->
      through_bool (fold_op output_size kernel_size stride dilation padding) x
  | _ -> fold_op output_size kernel_size stride dilation padding x

let matmul_op x y =
  if Intercept.intercepting () then Intercept.perform (Matmul (x, y))
  else Dispatch.matmul x y

let matmul (type a b) (x : (a, b) t) (y : (a, b) t) : (a, b) t =
  match dtype x with
  | Nx_dtype.Bit -> through_bool (fun x -> matmul_op x (bool y)) x
  | _ -> matmul_op x y

let fft ~inverse ~axes x =
  if Intercept.intercepting () then Intercept.perform (Fft { inverse; axes; x })
  else Dispatch.fft inverse axes x

let rfft dtype ~axes x =
  if Intercept.intercepting () then Intercept.perform (Rfft { dtype; axes; x })
  else Dispatch.rfft dtype axes x

let irfft ?s dtype ~axes x =
  if Intercept.intercepting () then
    Intercept.perform (Irfft { dtype; axes; s; x })
  else Dispatch.irfft dtype axes s x

let cholesky ~upper x =
  if Intercept.intercepting () then Intercept.perform (Cholesky { upper; x })
  else Dispatch.cholesky upper x

let qr ~reduced x = Intercept.eval (Qr { reduced; x })
let lu x = Intercept.eval (Lu x)
let svd ~full_matrices x = Intercept.eval (Svd { full_matrices; x })
let eigvals x = fst (Intercept.eval (Eig { vectors = false; x }))
let eigvalsh x = fst (Intercept.eval (Eigh { vectors = false; x }))

let with_vectors op = function
  | values, Some vectors -> (values, vectors)
  | _, None -> invalid_arg ("Nx." ^ op ^ ": the eigenvectors are missing")

let eig x = with_vectors "eig" (Intercept.eval (Eig { vectors = true; x }))
let eigh x = with_vectors "eigh" (Intercept.eval (Eigh { vectors = true; x }))

let solve_triangular ~upper ~transpose ~unit_diag a b =
  if Intercept.intercepting () then
    Intercept.perform (Solve_triangular { upper; transpose; unit_diag; a; b })
  else Dispatch.solve_triangular upper transpose unit_diag a b

let move x m =
  if Intercept.intercepting () then Intercept.perform (Move (x, m))
  else Dispatch.move x m

let reshape x shape = move x (Reshape shape)
let expand x shape = move x (Expand shape)
let permute x axes = move x (Permute axes)
let shrink x limits = move x (Shrink limits)
let flip x dims = move x (Flip dims)

let sliding_window x ~axis ~window ~step =
  move x (Window { axis; size = window; step })

(* The elements of [x]'s view in C order, in a host buffer, read by the surface
   function [by]: a host value's storage when they are one C-order run of it,
   starting on a byte. *)
let read ~by x =
  if Intercept.intercepting () then Intercept.perform (Read { by; x })
  else Dispatch.read_elements_of x

let check ok msg =
  if Intercept.intercepting () then Intercept.perform (Check { ok; msg })
  else Dispatch.check_elements ok msg

(* A value already at [p] is returned as it is. *)
let place (type a b) p (x : (a, b) t) : (a, b) t =
  if Placement.equal (placement x) p then
    match x with
    | Placed r -> Cell.with_borrow r.r_cell (fun () -> x)
    | Host _ | Traced _ -> x
  else if Intercept.intercepting () then Intercept.perform (Place (p, x))
  else Place.move_to p x

(* [copy x] is [x] in storage of its own, C-contiguous from its first element:
   it always copies. A traced value has no bytes: the interpretation that made
   it answers its copy. [contiguous x] is [x] itself when its view, a traced
   value's too, is already C-contiguous from its first element. *)
let copy x =
  if Intercept.intercepting () then Intercept.perform (Contiguous x)
  else Dispatch.copy x

let contiguous x =
  let v = view x in
  if View.is_c_contiguous v && View.offset v = 0 then x else copy x

(* Creation. A constant is one element on each device of its context, the host's
   included, expanded as a view to its shape: placing the element and moving it
   are views and copies, so a constant needs no kernel on any device, an
   interpretation sees a constant (a compiled call folds it into its kernels),
   and a kernel reads it with stride 0. A split context sees each device's
   window of the element expanded, a view of the same storage. [Nx.copy] gives a
   constant storage of its own. *)

let broadcast scalar shape_arr =
  if Array.length shape_arr = 0 then scalar
  else
    let ones = Array.map (fun _ -> 1) shape_arr in
    let x = reshape scalar ones in
    if Shape.equal ones shape_arr then x else expand x shape_arr

let full (ctx : context) dtype shape_arr value =
  if on_host ctx && not (Intercept.intercepting ()) then (
    (* The same expanded element, made without performing its movements: with no
       interpretation to see them, they are view arithmetic. *)
    let buffer = Elements.create dtype 1 in
    Elements.fill dtype buffer value;
    let strides = Array.make (Array.length shape_arr) 0 in
    Host { dtype; view = View.create ~strides shape_arr; buffer })
  else
    let copies =
      List.fold_left
        (fun p (axis, _) -> Placement.uncut p ~axis)
        ctx (Placement.cuts ctx)
    in
    let e = place copies (Host (filled dtype [||] value)) in
    place ctx (broadcast e shape_arr)

let from_host (ctx : context) dtype buffer =
  check_host "Nx.from_host" dtype buffer;
  let x =
    Host
      { dtype; view = View.create [| Nx_device.Buffer.length buffer |]; buffer }
  in
  if on_host ctx then x else place ctx x

(* The buffer of exactly [x]'s elements in C order, without a copy, when there
   is one: [x] is on the host, or its storage is one runtime buffer, and its
   view is a contiguous run of its storage. *)
let run (type a b) (x : (a, b) t) =
  match x with
  | Host t -> Place.run_in t.buffer t.view
  | Placed r -> (
      match Cell.state r.r_cell with
      | Live [ b ] -> Place.run_in b r.r_view
      | _ -> None)
  | Traced _ -> outside_trace ()

(* Values over runtime buffers *)

(* An empty buffer of [x]'s dtype on its device. *)
let empty (type a b) (x : (a, b) t) =
  let s = Nx_dtype.Scalar.of_dtype (dtype x) in
  match x with
  | Placed r when not (Placement.on_disk r.r_placement) ->
      Nx_device.Buffer.create
        (Device.memory (List.hd (Placement.devices r.r_placement)))
        s 0
  | Host _ | Placed _ | Traced _ -> Nx_device.Buffer.create Nx_device.host s 0

(* A value on the disk, which computes nothing, is copied to the host. *)
let to_buffer (type a b) (x : (a, b) t) =
  match x with
  | Traced _ -> invalid_arg "Nx.to_buffer: a traced value has no buffer"
  | Placed r
    when List.compare_length_with (Placement.devices r.r_placement) 1 > 0 ->
      invalid_arg
        (Format.asprintf "Nx.to_buffer: a value at %a is on several devices"
           Placement.pp r.r_placement)
  | Host _ | Placed _ -> (
      match run x with
      | Some b -> b
      | None when View.numel (view x) = 0 -> empty x
      | None -> (
          let y = copy x in
          match run y with Some b -> b | None -> read ~by:"Nx.to_buffer" y))
