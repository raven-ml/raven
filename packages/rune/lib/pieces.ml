(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module A = Nx_array
module L = Nx_array.Layout
module M = Nx_array.Move
module D = Nx.Dtype

exception Declined of string

let decline fmt = Printf.ksprintf (fun s -> raise (Declined s)) fmt
let shape_of (f : _ Nx.Prim.form) = Iarray.to_array (L.shape f.layout)
let extents w = Array.map (fun (r : M.range) -> r.count) w

let device_of p =
  let s = Nx.Placement.devices p in
  List.find_opt
    (fun r -> Nx.Placement.equal p (Nx.Placement.device s r))
    (Nx.rigs s)

(* Whether a program needs splitting *)

(* A placement and the device it holds a value on, if it is one. *)
type seen = Seen : 'd Nx.Placement.t * Rig.t option -> seen

let needed (p : Program.t) forms =
  let several = ref false and devices = ref [] in
  (* Most values of a program share a placement: the last one's device is
     kept rather than looked up again. *)
  let last = ref None in
  let device_of (type d) (q : d Nx.Placement.t) =
    match !last with
    | Some (Seen (q', r)) when Nx.Placement.equal q q' -> r
    | Some _ | None ->
        let r = device_of q in
        last := Some (Seen (q, r));
        r
  in
  let note placement =
    match device_of placement with
    | None -> several := true
    | Some r ->
        if not (List.exists (Rig.equal r) !devices) then
          devices := r :: !devices
  in
  let is_var x =
    match Nx.Prim.payload p.owner x with Some (Var _) -> true | _ -> false
  in
  (* A value over several devices needs splitting. Of the values on one device,
     the arguments, the results it computes and the placements say where the
     program computes: a creation elsewhere computes there too, and a constant
     elsewhere is placed, or copied, there. *)
  let operand (type v s d) (x : (v, s, d) Nx.t) =
    if Option.is_none (device_of (Nx.placement x)) then several := true
  in
  let any (Nx.Prim.Any x) = operand x in
  Array.iter (fun (F f) -> note f.placement) forms;
  Array.iter
    (fun (V y) -> if is_var y then note (Nx.placement y) else operand y)
    p.outputs;
  Array.iter
    (function
      | Op { op; _ } -> (
          let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
          List.iter any xs;
          match op with
          | Place (q, _) -> note q
          | Map { at = Some q; _ } ->
              if Option.is_none (device_of q) then several := true
          | _ -> ())
      | Do { c; _ } -> List.iter (fun (V x) -> operand x) (Program.operands c))
    p.eqns;
  !several || List.length !devices > 1

(* The engine

   A value over several devices is a stand-in of the splitting's own
   interpretation, holding its pieces; a value on one device is the tracer's
   variable, or a constant. Each step of a partition is an operation the tracer
   records, on one device, or a placement from one device onto another, a
   copy. *)

type ('v, 's, 'd) Nx.Prim.payload +=
  | Cut : ('v, 's, 'd) Nx.t iarray -> ('v, 's, 'd) Nx.Prim.payload
        (** The pieces, one per device of the placement, in order. *)

let rig_of p = Option.get (device_of p)

(* An empty value of [dt] and [shape] on the one device of [at]. *)
let empty (type v s e) (at : e Nx.Placement.t) (dt : (v, s) D.t) shape :
    (v, s, e) Nx.t =
  Nx.Repr.unsafe_of_array (Nx.Placement.devices at)
    (A.create (rig_of at) dt (Iarray.of_array shape))

let full x (w : M.range array) =
  let s = Nx.shape x in
  Array.length w = Array.length s
  && Array.for_all2
       (fun (r : M.range) n -> r.start = 0 && r.count = n && r.step = 1)
       w s

let bytes x = D.bytes (Nx.dtype x) (Array.fold_left ( * ) 1 (Nx.shape x))

(* The pieces of [x], a value of the splitting [ri] or a constant. *)
let pieces_of (type v s d) ri (x : (v, s, d) Nx.t) : (v, s, d) Nx.t iarray =
  match Nx.Prim.payload ri x with
  | Some (Cut ps) -> ps
  | _ -> (
      match device_of (Nx.placement x) with
      | Some _ -> Iarray.of_list [ x ]
      | None ->
          let s = Nx.Placement.devices (Nx.placement x) in
          Iarray.map (Nx.Repr.of_array s) (Nx.Repr.shards x))

let engine ri ~by moves : Nx.Prim.engine =
  let view mv x = Nx.Prim.eval ~by (Move (mv, x)) in
  let window x w =
    if full x w then x else view (M.Slice (Iarray.of_array w)) x
  in
  (* [x] placed at [at], a copy where it lies on another device. *)
  let put at x =
    let src = rig_of (Nx.placement x) and dst = rig_of at in
    if not (Rig.equal src dst) then
      moves :=
        Printf.sprintf "%s moves %d bytes from %s to %s" by (bytes x)
          (Rig.name src) (Rig.name dst)
        :: !moves;
    Nx.Prim.eval ~by (Place (at, x))
  in
  let bring at ps =
    let d = rig_of at in
    let here (x, _) = Rig.equal (rig_of (Nx.placement x)) d in
    let x, w =
      match List.find_opt here ps with Some p -> p | None -> List.hd ps
    in
    put at (window x w)
  in
  let gather (type v s d e) (at : e Nx.Placement.t) (dt : (v, s) D.t) s
      (boxes : (M.range array * (v, s, d) Nx.t * M.range array) list) :
      (v, s, e) Nx.t =
    let pieces =
      List.map (fun (w, x, b) -> (Iarray.of_array w, put at (window x b))) boxes
    in
    Nx.Prim.eval ~by
      (Assemble { dtype = dt; shape = s; fill = D.zero dt; pieces })
  in
  let assemble (type v s d e) (p : e Nx.Placement.t) s (dt : (v, s) D.t)
      (ps : (v, s, d) Nx.t option array) : (v, s, e) Nx.t =
    let set = Nx.Placement.devices p in
    let windows = Nx.Prim.pieces p s in
    let piece j =
      match ps.(j) with
      | Some v -> Option.get (Nx.Prim.coerce dt set v)
      | None ->
          let here, w = windows.(j) in
          empty here dt (extents w)
    in
    if Array.length windows = 1 then piece 0
    else
      let form =
        {
          Nx.Prim.dtype = dt;
          layout = L.contiguous (Iarray.of_array s);
          placement = p;
        }
      in
      Nx.Prim.traced ri form (Cut (Iarray.init (Array.length windows) piece))
  in
  {
    run = (fun ~by ?into:_ _ _ op -> Nx.Prim.eval ~by op);
    piece = (fun x j _ -> Iarray.get (pieces_of ri x) j);
    view;
    bring;
    gather;
    assemble;
  }

(* Splitting *)

type output = One of int | Over of form * int array

type t = {
  program : Program.t;
  forms : form array;
  args : int array;
  outputs : output array;
  moves : string list;
}

(* The forms of the pieces of a value of [f]: [f] itself on one device. *)
let piece_forms (F f) =
  match device_of f.placement with
  | Some _ -> [| F f |]
  | None ->
      Array.map
        (fun (here, w) ->
          F
            {
              Nx.Prim.dtype = f.dtype;
              layout = L.contiguous (Iarray.of_array (extents w));
              placement = here;
            })
        (Nx.Prim.pieces f.placement (shape_of f))

let split ~name (p : Program.t) forms =
  if Array.length p.adds > 0 then decline "a total over several devices";
  if p.emits then decline "an emission over several devices";
  let leaves = Array.map piece_forms forms in
  let moves = ref [] and outputs = ref [||] in
  let refuse _ ~by _ =
    invalid_arg (by ^ ": a value over several devices outside its partition")
  in
  Nx.Prim.interpret ~name Values refuse @@ fun ri ->
  let body vs =
    let env = Array.make p.slots None and at = ref 0 in
    Array.iteri
      (fun k (F f) ->
        let n = Array.length leaves.(k) in
        let ps = Array.sub vs !at n in
        at := !at + n;
        env.(p.params.(k)) <-
          Some
            (if n = 1 && Option.is_some (device_of f.placement) then ps.(0)
             else
               let set = Nx.Placement.devices f.placement in
               let cut =
                 Iarray.map (Program.cast f.dtype set) (Iarray.of_array ps)
               in
               V (Nx.Prim.traced ri f (Cut cut))))
      forms;
    let lookup (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
      match Nx.Prim.payload p.owner x with
      | Some (Var n) -> like x (Option.get env.(n))
      | _ -> x
    in
    let bind = ref 0 in
    let store (type v s d) (y : (v, s, d) Nx.t) =
      env.(!bind) <- Some (V y);
      incr bind
    in
    Array.iter
      (function
        | Op { by; op; base } ->
            let op = Nx.Prim.map lookup op in
            let r =
              match op with
              | Check _ -> decline "a check over several devices"
              | Place (q, x)
                when Option.is_some (Nx.Prim.owner x)
                     && Option.is_none
                          (Nx.Prim.coerce (Nx.dtype x) (Nx.Placement.devices q)
                             x) ->
                  decline "Nx.place onto another set of devices"
              | _ -> Nx.Prim.partition (engine ri ~by moves) ~by op
            in
            bind := base;
            Nx.Prim.iter_results store op r
        | Do { c = Detach x; base } -> env.(base) <- Some (V (lookup x))
        | Do { c; _ } -> decline "%s over several devices" (construct_name c))
      p.eqns;
    let pieces =
      Array.map
        (fun (V y) ->
          let y = lookup y in
          Array.map (fun v -> V v) (Iarray.to_array (pieces_of ri y)))
        p.outputs
    in
    let next = ref 0 in
    outputs :=
      Array.mapi
        (fun j ps ->
          let k = !next in
          next := k + Array.length ps;
          let (F f) = form p.outputs.(j) in
          match device_of f.placement with
          | Some _ -> One k
          | None -> Over (F f, Array.init (Array.length ps) (( + ) k)))
        pieces;
    Array.concat (Array.to_list pieces)
  in
  let forms = Array.concat (Array.to_list leaves) in
  let program, _ = Trace.trace Jit ~name forms body in
  {
    program;
    forms;
    args = Array.map Array.length leaves;
    outputs = !outputs;
    moves = List.rev !moves;
  }
