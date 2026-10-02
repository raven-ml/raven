(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array
open Value

(* Dispatch

   With no interpretation, operands all on the host run nx.cpu directly, and any
   other operands run on each device of the placement their route gives, the one
   their placed operands share, by that device's backend. Operands at another
   placement run once per device, on that device's arrays, after nx copies there
   the operands that are not: host values, values on the disk, and values that
   the device needs whole where they are split. nx answers movements, placing
   and reading itself. *)

let all_host xs = List.for_all (function Host _ -> true | _ -> false) xs

(* How each device's results become the operation's: [settle] makes a value of
   one result's arrays, one per device. *)
type settle = { settle : 'a 'b. ('a, 'b) Nx_array.t list -> ('a, 'b) t }

let host_settle =
  {
    settle =
      (function [ a ] -> Host a | _ -> invalid_arg "Nx: one host result");
  }

(* A kernel that refuses its operands raises [Refused] to nx, which names the
   backend, the device, the operation and the remedies: nothing falls through
   to another backend. *)
let refused op (e : Kernels.env) reason =
  let (module K) = e.kernels in
  invalid_arg
    (Printf.sprintf
       "Nx.%s: %s on %s has no kernel: %s. Compile it with Rune.jit, or place \
        the operands elsewhere."
       op K.name
       (Nx_device.name (Device.memory e.device))
       reason)

(* [compute envs s op] is [op] computed where each of [envs] says, its results
   made by [s]. *)
let compute : type r. Kernels.env list -> settle -> r Op.t -> r =
 fun envs s op ->
  let on e f =
    match f e with
    | v -> v
    | exception Nx_backend.Refused reason -> refused (Op.name op) e reason
  in
  let envs_map f = List.map (fun e -> on e f) envs in
  let each f = s.settle (envs_map f) in
  match[@warning "@4@8"] op with
  | Unary (k, x) -> each (fun e -> Kernels.unary e k (e.arr x))
  | Binary (k, a, b) -> each (fun e -> Kernels.binary e k (e.arr a) (e.arr b))
  | Compare (k, a, b) -> each (fun e -> Kernels.compare e k (e.arr a) (e.arr b))
  | Where (c, a, b) ->
      each (fun e -> Kernels.where e (e.arr c) (e.arr a) (e.arr b))
  | Fma (a, b, c) -> each (fun e -> Kernels.fma e (e.arr a) (e.arr b) (e.arr c))
  | Reduce (k, axes, x) -> each (fun e -> Kernels.reduce e k axes (e.arr x))
  | Scan (k, axis, x) -> each (fun e -> Kernels.scan e k axis (e.arr x))
  | Arg_reduce (k, axis, x) ->
      each (fun e -> Kernels.arg_reduce e k axis (e.arr x))
  | Sort { descending; axis; x } ->
      each (fun e -> Kernels.sort e descending axis (e.arr x))
  | Argsort { descending; axis; x } ->
      each (fun e -> Kernels.argsort e descending axis (e.arr x))
  | Group { x; _ } -> each (fun e -> Kernels.group e (e.arr x))
  | Pad (padding, v, x) -> each (fun e -> Kernels.pad e padding v (e.arr x))
  | Cat (axis, xs) -> each (fun e -> Kernels.cat e axis (List.map e.arr xs))
  | Convert (Cast, dtype, x) -> each (fun e -> Kernels.cast e dtype (e.arr x))
  | Convert (Bitcast, dtype, x) ->
      each (fun e -> Kernels.bitcast e dtype (e.arr x))
  | Threefry (key, ctr) ->
      each (fun e -> Kernels.threefry e (e.arr key) (e.arr ctr))
  | Gather (axis, indices, data) ->
      each (fun e -> Kernels.gather e axis (e.arr indices) (e.arr data))
  | Scatter { mode; unique; axis; indices; updates; into } ->
      each (fun e ->
          Kernels.scatter e mode unique axis (e.arr indices) (e.arr updates)
            (e.arr into))
  | Update (x, starts, v) ->
      each (fun e -> Kernels.update e (e.arr x) (e.arr starts) (e.arr v))
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      each (fun e ->
          Kernels.unfold e kernel_size stride dilation padding (e.arr x))
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      each (fun e ->
          Kernels.fold e output_size kernel_size stride dilation padding
            (e.arr x))
  | Matmul (a, b) -> each (fun e -> Kernels.matmul e (e.arr a) (e.arr b))
  | Fft { inverse; axes; x } ->
      each (fun e -> Kernels.fft e inverse axes (e.arr x))
  | Rfft { dtype; axes; x } ->
      each (fun e -> Kernels.rfft e dtype axes (e.arr x))
  | Irfft { dtype; axes; s = sizes; x } ->
      each (fun e -> Kernels.irfft e dtype axes sizes (e.arr x))
  | Contiguous x -> each (fun e -> Kernels.contiguous e (e.arr x))
  | Cholesky { upper; x } -> each (fun e -> Kernels.cholesky e upper (e.arr x))
  | Qr { reduced; x } ->
      let rs = envs_map (fun e -> Kernels.qr e reduced (e.arr x)) in
      (s.settle (List.map fst rs), s.settle (List.map snd rs))
  | Lu x ->
      let rs = envs_map (fun e -> Kernels.lu e (e.arr x)) in
      ( s.settle (List.map (fun (a, _, _) -> a) rs),
        s.settle (List.map (fun (_, a, _) -> a) rs),
        s.settle (List.map (fun (_, _, a) -> a) rs) )
  | Svd { full_matrices; x } ->
      let rs = envs_map (fun e -> Kernels.svd e full_matrices (e.arr x)) in
      ( s.settle (List.map (fun (a, _, _) -> a) rs),
        s.settle (List.map (fun (_, a, _) -> a) rs),
        s.settle (List.map (fun (_, _, a) -> a) rs) )
  | Eig { vectors; x } ->
      let rs = envs_map (fun e -> Kernels.eig e vectors (e.arr x)) in
      ( s.settle (List.map fst rs),
        if vectors then
          Some (s.settle (List.map (fun (_, v) -> Option.get v) rs))
        else None )
  | Eigh { vectors; x } ->
      let rs = envs_map (fun e -> Kernels.eigh e vectors (e.arr x)) in
      ( s.settle (List.map fst rs),
        if vectors then
          Some (s.settle (List.map (fun (_, v) -> Option.get v) rs))
        else None )
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      each (fun e ->
          Kernels.solve_triangular e upper transpose unit_diag (e.arr a)
            (e.arr b))
  | Move _ | Place _ | Read _ | Check _ ->
      invalid_arg "Nx: the operation computes nothing"

(* The results of every device at [q], each the whole result, or, when
   [windowed], each device's window of the whole result copied out on it. *)
let settle_on q ~windowed envs =
  let ds = Placement.devices q in
  let window e d a =
    Placement.check_shape "Nx" q (Kernels.shape_of a);
    let w = Placement.window q (Kernels.shape_of a) d in
    Kernels.contiguous e { a with view = View.shrink a.view w }
  in
  {
    settle =
      (fun arrays ->
        let arrays =
          if windowed then
            List.map2
              (fun (e, d) a -> window e d a)
              (List.combine envs ds) arrays
          else arrays
        in
        let a = List.hd arrays in
        placed "Nx" q a.dtype
          (View.create (Kernels.shape_of a))
          (cell ~placement:q
             ~length:(Nx_device.Buffer.length a.buffer)
             (List.map (fun (a : (_, _) Nx_array.t) -> a.buffer) arrays)));
  }

(* Whether each device can compute its tile of the result at [q] from its tiles
   of the operands, split alike: a gather reads its data whole along its
   axis. *)
let per_tile q : Route.rule -> bool = function
  | Elementwise | Along _ | Reduce _ -> true
  | Gather axis -> not (List.mem_assoc axis (Placement.cuts q))
  | Contract | Into -> false

(* An eager operation on a device without a backend raises before any work: a
   compiled function, a backend paired with the device, or the host computes
   it. *)
let no_kernels op d =
  invalid_arg
    (Printf.sprintf
       "Nx.%s: %s has no eager kernels. Compile it with Rune.jit, pair the \
        device with a backend (Nx.Device.with_backend), or place the operands \
        on Nx.Placement.host."
       op (Device.name d))

let rec with_cells cells f =
  match cells with
  | [] -> f ()
  | c :: rest -> Cell.with_borrow c (fun () -> with_cells rest f)

let cells_of op =
  List.fold_left
    (fun cells (P x) ->
      match x with
      | Placed r when not (List.memq r.r_cell cells) -> r.r_cell :: cells
      | _ -> cells)
    [] (Op.operands op)

(* [on_devices op] is [op] computed where it runs: over host operands by nx.cpu,
   and at a placement [q] once per device of [q], by its backend. Each device
   holds its tiles of the operands when the result is split and the operation
   keeps tiles apart, and whole copies of them otherwise, which nx copies there
   first; a split result is then each device's window of the whole it computed.
   A contraction split along an outer axis of its left operand is computed whole
   on each device too, N times the work of a tile each: one rule serves every
   split contraction and scatter until a consumer needs the tiles. *)
let on_devices : type r. r Op.t -> r =
 fun op ->
  match Route.routing op with
  | Computes (rule, xs) -> (
      match
        Route.route (fun (P x) -> Route.placement_of x) (Op.name op) rule xs
      with
      | On_host -> compute [ Kernels.host ] host_settle op
      | At q ->
          let ds = Placement.devices q in
          let kernels =
            List.map
              (fun d ->
                match d.Device.backend with
                | Some k -> k
                | None -> no_kernels (Op.name op) d)
              ds
          in
          let split =
            List.find_map
              (fun (P x) ->
                match Route.placement_of x with
                | Some p when Placement.cuts p <> [] -> Some p
                | _ -> None)
              xs
          in
          let cut = Placement.cuts q <> [] in
          let target, windowed =
            match split with
            | Some g when cut && per_tile q rule -> (g, false)
            | _ ->
                let whole =
                  List.fold_left
                    (fun g (a, _) -> Placement.uncut g ~axis:a)
                    q (Placement.cuts q)
                in
                (whole, cut)
          in
          let op =
            Op.map_operands { f = (fun x -> Place.move_to target x) } op
          in
          let envs = List.map2 Kernels.on kernels ds in
          with_cells (cells_of op) (fun () ->
              compute envs (settle_on q ~windowed envs) op))
  | Moves _ | Places _ | Reads _ ->
      invalid_arg "Nx: the operation computes nothing"

(* Each operation with no interpretation: nx.cpu on host operands, with no
   closure and no operation built, and [on_devices] otherwise. *)

let unary k x =
  match x with
  | Host a -> Host (Kernels.unary Kernels.host k a)
  | _ -> on_devices (Unary (k, x))

let binary k x y =
  match (x, y) with
  | Host a, Host b -> Host (Kernels.binary Kernels.host k a b)
  | _ -> on_devices (Binary (k, x, y))

let compare k x y =
  match (x, y) with
  | Host a, Host b -> Host (Kernels.compare Kernels.host k a b)
  | _ -> on_devices (Compare (k, x, y))

let where c x y =
  match (c, x, y) with
  | Host c', Host a, Host b -> Host (Kernels.where Kernels.host c' a b)
  | _ -> on_devices (Where (c, x, y))

let fma a b c =
  match (a, b, c) with
  | Host a, Host b, Host c -> Host (Kernels.fma Kernels.host a b c)
  | _ -> on_devices (Fma (a, b, c))

let reduce k axes x =
  match x with
  | Host a -> Host (Kernels.reduce Kernels.host k axes a)
  | _ -> on_devices (Reduce (k, axes, x))

let matmul x y =
  match (x, y) with
  | Host a, Host b -> Host (Kernels.matmul Kernels.host a b)
  | _ -> on_devices (Matmul (x, y))

let copy x =
  match x with
  | Host a -> Host (Kernels.contiguous Kernels.host a)
  | _ -> on_devices (Contiguous x)

(* [x] moved by [m]. A reshape that [x]'s strides cannot express, such as a
   flattened transpose, moves a C-order copy of [x]. *)
let move x m =
  match m with
  | Move.Reshape s when not (View.can_reshape (view x) s) ->
      Move.apply (copy x) m
  | _ -> Move.apply x m

let scan k axis x =
  match x with
  | Host a -> Host (Kernels.scan Kernels.host k axis a)
  | _ -> on_devices (Scan (k, axis, x))

let arg_reduce k axis x =
  match x with
  | Host a -> Host (Kernels.arg_reduce Kernels.host k axis a)
  | _ -> on_devices (Arg_reduce (k, axis, x))

let sort descending axis x =
  match x with
  | Host a -> Host (Kernels.sort Kernels.host descending axis a)
  | _ -> on_devices (Sort { descending; axis; x })

let argsort descending axis x =
  match x with
  | Host a -> Host (Kernels.argsort Kernels.host descending axis a)
  | _ -> on_devices (Argsort { descending; axis; x })

let group by x =
  match x with
  | Host a -> Host (Kernels.group Kernels.host a)
  | _ -> on_devices (Group { by; x })

let pad padding v x =
  match x with
  | Host a -> Host (Kernels.pad Kernels.host padding v a)
  | _ -> on_devices (Pad (padding, v, x))

let cat axis xs =
  if all_host xs then
    Host (Kernels.cat Kernels.host axis (List.map Place.host_of xs))
  else on_devices (Cat (axis, xs))

let convert (type a b c d) (c : Op.conversion) (dtype : (c, d) Nx_dtype.t)
    (x : (a, b) t) : (c, d) t =
  match (c, x) with
  | Cast, Host a -> Host (Kernels.cast Kernels.host dtype a)
  | Bitcast, Host a -> Host (Kernels.bitcast Kernels.host dtype a)
  | _ -> on_devices (Convert (c, dtype, x))

let threefry key ctr =
  match (key, ctr) with
  | Host k, Host c -> Host (Kernels.threefry Kernels.host k c)
  | _ -> on_devices (Threefry (key, ctr))

let gather axis indices data =
  match (data, indices) with
  | Host d, Host i -> Host (Kernels.gather Kernels.host axis i d)
  | _ -> on_devices (Gather (axis, indices, data))

let scatter mode unique axis indices updates into =
  match (into, indices, updates) with
  | Host d, Host i, Host u ->
      Host (Kernels.scatter Kernels.host mode unique axis i u d)
  | _ -> on_devices (Scatter { mode; unique; axis; indices; updates; into })

let update x starts v =
  match (x, starts, v) with
  | Host a, Host s, Host w -> Host (Kernels.update Kernels.host a s w)
  | _ -> on_devices (Update (x, starts, v))

let unfold kernel_size stride dilation padding x =
  match x with
  | Host a ->
      Host (Kernels.unfold Kernels.host kernel_size stride dilation padding a)
  | _ -> on_devices (Unfold { kernel_size; stride; dilation; padding; x })

let fold output_size kernel_size stride dilation padding x =
  match x with
  | Host a ->
      Host
        (Kernels.fold Kernels.host output_size kernel_size stride dilation
           padding a)
  | _ ->
      on_devices
        (Fold { output_size; kernel_size; stride; dilation; padding; x })

let fft inverse axes x =
  match x with
  | Host a -> Host (Kernels.fft Kernels.host inverse axes a)
  | _ -> on_devices (Fft { inverse; axes; x })

let rfft dtype axes x =
  match x with
  | Host a -> Host (Kernels.rfft Kernels.host dtype axes a)
  | _ -> on_devices (Rfft { dtype; axes; x })

let irfft dtype axes s x =
  match x with
  | Host a -> Host (Kernels.irfft Kernels.host dtype axes s a)
  | _ -> on_devices (Irfft { dtype; axes; s; x })

let cholesky upper x =
  match x with
  | Host a -> Host (Kernels.cholesky Kernels.host upper a)
  | _ -> on_devices (Cholesky { upper; x })

let solve_triangular upper transpose unit_diag a b =
  match (a, b) with
  | Host x, Host y ->
      Host (Kernels.solve_triangular Kernels.host upper transpose unit_diag x y)
  | _ -> on_devices (Solve_triangular { upper; transpose; unit_diag; a; b })

(* Exactly the elements of [x]'s view in C order, in a host buffer: a host
   value's storage when they are one run of it, gathered by nx.cpu under a read
   claim otherwise, and a copy of a placed one's, read from its runtime
   buffers. *)
let read_elements_of (type a b) (x : (a, b) t) : Nx_device.Buffer.t =
  match x with
  | Host t -> (
      match Place.run_in t.buffer t.view with
      | Some b -> b
      | None -> (Kernels.contiguous Kernels.host t).buffer)
  | Placed r -> Place.read_elements r
  | Traced _ -> outside_trace ()

(* [check_elements ok msg] raises [Invalid_argument (msg i)] for the index [i]
   of [ok]'s first false element in C order, its elements read under a read
   claim. *)
let check_elements ok msg =
  let shape = View.shape (view ok) in
  let n = Shape.numel shape in
  if n > 0 then begin
    let b = read_elements_of ok in
    Nx_device.Buffer.Claim.read b;
    let get = Elements.get Nx_dtype.bool b in
    let rec first i = if i = n || not (get i) then i else first (i + 1) in
    let i =
      Fun.protect
        ~finally:(fun () -> Nx_device.Buffer.Claim.release b)
        (fun () -> first 0)
    in
    if i < n then invalid_arg (msg (Shape.unravel_index i shape))
  end

(* [direct op] answers [op] with no interpretation. The decompositions run
   through [on_devices] even on the host, which settles each result. *)
let direct : type r. r Op.t -> r =
 fun op ->
  match[@warning "@4@8"] op with
  | Unary (k, x) -> unary k x
  | Binary (k, x, y) -> binary k x y
  | Compare (k, x, y) -> compare k x y
  | Where (c, x, y) -> where c x y
  | Fma (a, b, c) -> fma a b c
  | Reduce (k, axes, x) -> reduce k axes x
  | Scan (k, axis, x) -> scan k axis x
  | Arg_reduce (k, axis, x) -> arg_reduce k axis x
  | Sort { descending; axis; x } -> sort descending axis x
  | Argsort { descending; axis; x } -> argsort descending axis x
  | Group { by; x } -> group by x
  | Pad (padding, v, x) -> pad padding v x
  | Cat (axis, xs) -> cat axis xs
  | Convert (c, dtype, x) -> convert c dtype x
  | Threefry (key, ctr) -> threefry key ctr
  | Gather (axis, indices, data) -> gather axis indices data
  | Scatter { mode; unique; axis; indices; updates; into } ->
      scatter mode unique axis indices updates into
  | Update (x, starts, v) -> update x starts v
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      unfold kernel_size stride dilation padding x
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      fold output_size kernel_size stride dilation padding x
  | Matmul (x, y) -> matmul x y
  | Fft { inverse; axes; x } -> fft inverse axes x
  | Rfft { dtype; axes; x } -> rfft dtype axes x
  | Irfft { dtype; axes; s; x } -> irfft dtype axes s x
  | Contiguous x -> copy x
  | Cholesky { upper; x } -> cholesky upper x
  | Qr _ | Lu _ | Svd _ | Eig _ | Eigh _ -> on_devices op
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      solve_triangular upper transpose unit_diag a b
  | Move (x, m) -> move x m
  | Place (p, x) -> Place.move_to p x
  | Read { x; _ } -> read_elements_of x
  | Check { ok; msg } -> check_elements ok msg
