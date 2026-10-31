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

let cpu = Ops.Single "CPU"
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
let bind (type v s d) ~direct bound (x : (v, s, d) Nx.t) =
  let dt = Nx.dtype x and shape = Nx.shape x in
  let tdt = L.dtype (D.Any dt) and k = parts dt in
  let leaf bytes n =
    let u = Ops.new_buffer cpu (max 1 n) tdt in
    bound := (u, [ host_buffer bytes ]) :: !bound;
    u
  in
  match Nx.Repr.array x with
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

(* Lowering *)

let rule ~direct bound i ~by (type r) (op : r Nx.Prim.t) : r =
  let node : type v s d. (v, s, d) Nx.t -> Ops.t =
   fun x ->
    match Nx.Prim.payload i x with
    | Some (Node u) -> u
    | Some _ -> invalid_arg "Lowered: a value of another payload"
    | None -> bind ~direct bound x
  in
  let ys = Rune.Lower.op ~device:cpu ~host:true node op in
  Nx.Prim.results ~by (fun k f -> Nx.Prim.traced i f (Node ys.(k))) op

(* [f ()]'s graph, the storage it binds and its result's elements. *)
let lowered (type v s d) ?(direct = false) (f : unit -> (v, s, d) Nx.t) =
  let bound = ref [] in
  let y, u =
    Nx.Prim.interpret ~name:"Lowered" Nx.Prim.Extent
      (fun i ~by op -> rule ~direct bound i ~by op)
      (fun i ->
        let y = f () in
        match Nx.Prim.payload i y with
        | Some (Node u) -> (y, u)
        | _ -> invalid_arg "Lowered: the result is not lowered")
  in
  let dt = (Nx.Prim.form y).dtype and shape = Nx.shape y in
  (bound, u, dt, shape, numel shape * parts dt)

(* The compiled schedule storing [u], of [n] elements, into [o]. *)
let compiled u o n =
  let sink =
    Ops.sink [ Ops.after o [ Ops.store o (Shape.reshape u [ Ops.Int n ]) ] ]
  in
  let linear, vars = Schedule.create_linear_with_vars sink in
  let targets _ = Tolk_engine.target Rig.host in
  (Realize.compile_linear ~targets linear, vars)

let run (type v s d) (f : unit -> (v, s, d) Nx.t) : v array =
  let bound, u, dt, shape, n = lowered f in
  let tdt = L.dtype (D.Any dt) in
  if n = 0 then [||]
  else
    let o = Ops.new_buffer cpu n tdt in
    let compiled, vars = compiled u o n in
    let len = n * Dtype.itemsize tdt in
    let out = Rig.Buffer.create Rig.host len in
    let s =
      Tolk_engine.link
        ~devices:(fun _ -> Rig.host)
        ~bound:((o, [ out ]) :: !bound) compiled
    in
    Tolk_engine.run ~vars s [||];
    let bytes = Bytes.create len in
    Rig.Buffer.blit_to_bytes out 0 bytes 0 len;
    decode dt shape bytes

let work (type v s d) (f : unit -> (v, s, d) Nx.t) =
  let _, u, dt, _, n = lowered ~direct:true f in
  let o = Ops.new_buffer cpu (max 1 n) (L.dtype (D.Any dt)) in
  let compiled, vars = compiled u o (max 1 n) in
  let count x = Shape.sym_infer x vars in
  List.fold_left
    (fun (ops, lds) call ->
      if not (Op.equal (Ops.op call) Op.Call) then
        invalid_arg "Lowered.work: a schedule with loops";
      let e = Realize.estimate_uop call in
      (ops + count e.ops, lds + count e.lds))
    (0, 0) (Ops.src compiled)
