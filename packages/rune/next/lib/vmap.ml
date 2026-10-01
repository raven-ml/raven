(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Repr = Nx.Repr

type t = {
  entry : string;
  axis : Construct.axis option;
  size : int;
  id : unit ref;  (** The installation's identity, which its lanes name. *)
}

let create ?axis entry size = { entry; axis; size; id = ref () }

type (_, _) Repr.node +=
  | Lane : { map : t; batched : ('a, 'b) Nx.t } -> ('a, 'b) Repr.node

let lane m x =
  let s = Nx.shape x in
  Repr.Traced.v ~context:(Repr.context x)
    (Nx.Placement.without_leading_axis (Nx.placement x))
    (Nx.dtype x)
    (Array.sub s 1 (Array.length s - 1))
    (Lane { map = m; batched = x })

let owns m x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Lane { map; _ } -> map.id == m.id
      | _ -> false)
  | Host _ | Placed _ -> false

(* [physical m x] is the batched tensor of [x] if it is a lane of [m], and [x]
   otherwise. *)
let physical (type a b) m (x : (a, b) Nx.t) : (a, b) Nx.t =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Lane { map; batched } when map.id == m.id -> batched
      | _ -> x)
  | Host _ | Placed _ -> x

(* [batched m x] is [x] with the map's axis in front: a lane's batched tensor,
   or a value every lane shares broadcast along a new leading axis. *)
let batched m x =
  if owns m x then physical m x
  else
    let s = Nx.shape x in
    Nx.broadcast_to
      (Array.append [| m.size |] s)
      (Nx.reshape (Array.append [| 1 |] s) x)

(* The sum over the lanes of a value each lane holds. *)
let sum_lanes m v =
  if owns m v then Nx.sum ~axes:[ 0 ] (physical m v)
  else Nx.mul_s v (Nx_dtype.of_float (Nx.dtype v) (Float.of_int m.size))

let named m axis =
  match m.axis with Some a -> Type.Id.uid a = Type.Id.uid axis | None -> false

(* An axis counted in a lane's shape, counted in its batched tensor's. *)
let shifted axis = axis + 1

(* Batching *)

let move m x mv =
  let p = physical m x in
  match[@warning "@4@8"] mv with
  | Reshape s ->
      let p = if Nx.is_c_contiguous p then p else Nx.contiguous p in
      eval (Move (p, Reshape (Array.append [| m.size |] s)))
  | Expand s ->
      eval
        (Move
           ( Nx.reshape
               (Array.append [| m.size |]
                  (Array.append
                     (Array.make (Array.length s - Nx.ndim x) 1)
                     (Nx.shape x)))
               p,
             Expand (Array.append [| m.size |] s) ))
  | Permute axes ->
      eval (Move (p, Permute (Array.append [| 0 |] (Array.map (( + ) 1) axes))))
  | Shrink limits ->
      eval (Move (p, Shrink (Array.append [| (0, m.size) |] limits)))
  | Flip dims -> eval (Move (p, Flip (Array.append [| false |] dims)))
  | Window w -> eval (Move (p, Window { w with axis = shifted w.axis }))

(* A window written per lane at its own [starts]: along each axis, the rows of
   [v] at the lane's offset, kept where they fall inside the window. *)
let update m x starts v =
  let x = batched m x and v = batched m v in
  if owns m starts then begin
    let starts = physical m starts in
    let shape = Array.sub (Nx.shape x) 1 (Nx.ndim x - 1)
    and vshape = Array.sub (Nx.shape v) 1 (Nx.ndim v - 1) in
    let rank = Array.length shape in
    let window = ref v and inside = ref None in
    for axis = 0 to rank - 1 do
      let n = shape.(axis) and len = vshape.(axis) in
      let start =
        Nx.reshape [| m.size; 1 |] (Nx.slice [ Nx.A; Nx.I axis ] starts)
      in
      let rel =
        Nx.sub (Nx.reshape [| 1; n |] (Nx.arange Nx.int32 0 n 1)) start
      in
      let within =
        Nx.logical_and
          (Nx.greater_equal_s rel 0l)
          (Nx.less_s rel (Int32.of_int len))
      in
      let index = Nx.clamp ~min:0l ~max:(Int32.of_int (len - 1)) rel in
      let along =
        Array.init (rank + 1) (fun d ->
            if d = 0 then m.size else if d = axis + 1 then n else 1)
      in
      let target =
        Array.mapi (fun d s -> if d = axis + 1 then n else s) (Nx.shape !window)
      in
      window :=
        Nx.take_along_axis ~axis:(axis + 1)
          ~indices:(Nx.broadcast_to target (Nx.reshape along index))
          !window;
      let within = Nx.reshape along within in
      inside :=
        Some
          (match !inside with
          | None -> within
          | Some w -> Nx.logical_and w within)
    done;
    match !inside with None -> v | Some inside -> Nx.where inside !window x
  end
  else eval (Update (x, Nx.pad [| (1, 0) |] 0l starts, v))

(* A product broadcasts its operands' leading axes positionally, so an operand
   with leading axes of its own lifts both to one leading rank, the map's axis
   first. *)
let matmul m a b =
  let lead x = Nx.ndim x - 2 in
  let l = Int.max (lead a) (lead b) in
  if l = 0 then eval (Matmul (physical m a, physical m b))
  else
    let lift x =
      let s = Nx.shape x in
      let s = Array.append (Array.make (l - lead x) 1) s in
      if owns m x then Nx.reshape (Array.append [| m.size |] s) (physical m x)
      else Nx.reshape (Array.append [| 1 |] s) x
    in
    eval (Matmul (lift a, lift b))

(* [run m op] is [op], one of whose operands is a lane of [m], as one operation
   on the batched tensors: the map's axis in front, shapes gaining it and axes
   shifted past it, a value the lanes share broadcast along it. *)
let run : type r. t -> r Nx.Op.t -> r =
 fun m op ->
  let p x = physical m x and b x = batched m x and lane x = lane m x in
  let axes = Array.map shifted in
  match[@warning "@4@8"] op with
  | Unary (k, x) -> lane (eval (Unary (k, p x)))
  | Binary (k, x, y) -> lane (eval (Binary (k, b x, b y)))
  | Compare (k, x, y) -> lane (eval (Compare (k, b x, b y)))
  | Where (c, x, y) -> lane (eval (Where (b c, b x, b y)))
  | Reduce (k, a, x) -> lane (eval (Reduce (k, axes a, p x)))
  | Scan (k, axis, x) -> lane (eval (Scan (k, shifted axis, p x)))
  | Arg_reduce (k, axis, x) -> lane (eval (Arg_reduce (k, shifted axis, p x)))
  | Sort s -> lane (eval (Sort { s with axis = shifted s.axis; x = p s.x }))
  | Argsort s ->
      lane (eval (Argsort { s with axis = shifted s.axis; x = p s.x }))
  | Pad (padding, v, x) ->
      lane (eval (Pad (Array.append [| (0, 0) |] padding, v, p x)))
  | Cat (axis, xs) -> lane (eval (Cat (shifted axis, List.map b xs)))
  | Convert (k, dtype, x) -> lane (eval (Convert (k, dtype, p x)))
  | Threefry (key, ctr) -> lane (eval (Threefry (b key, b ctr)))
  | Gather (axis, indices, x) ->
      lane (eval (Gather (shifted axis, b indices, b x)))
  | Scatter s ->
      lane
        (eval
           (Scatter
              {
                s with
                axis = shifted s.axis;
                indices = b s.indices;
                updates = b s.updates;
                into = b s.into;
              }))
  | Update (x, starts, v) -> lane (update m x starts v)
  | Unfold u -> lane (eval (Unfold { u with x = p u.x }))
  | Fold f -> lane (eval (Fold { f with x = p f.x }))
  | Matmul (x, y) -> lane (matmul m x y)
  | Fft f -> lane (eval (Fft { f with axes = axes f.axes; x = p f.x }))
  | Rfft f -> lane (eval (Rfft { f with axes = axes f.axes; x = p f.x }))
  | Irfft f -> lane (eval (Irfft { f with axes = axes f.axes; x = p f.x }))
  | Contiguous x -> lane (eval (Contiguous (p x)))
  | Cholesky c -> lane (eval (Cholesky { c with x = p c.x }))
  | Qr q ->
      let q, r = eval (Qr { q with x = p q.x }) in
      (lane q, lane r)
  | Lu x ->
      let packed, pivots, perm = eval (Lu (p x)) in
      (lane packed, lane pivots, lane perm)
  | Svd s ->
      let u, sv, vt = eval (Svd { s with x = p s.x }) in
      (lane u, lane sv, lane vt)
  | Eig e ->
      let values, vectors = eval (Eig { e with x = p e.x }) in
      (lane values, Option.map lane vectors)
  | Eigh e ->
      let values, vectors = eval (Eigh { e with x = p e.x }) in
      (lane values, Option.map lane vectors)
  | Solve_triangular s ->
      lane (eval (Solve_triangular { s with a = b s.a; b = b s.b }))
  | Move (x, mv) -> lane (move m x mv)
  | Place (q, x) -> lane (eval (Place (Nx.Placement.with_leading_axis q, p x)))
  | Read _ ->
      invalid_arg
        "Rune: cannot read the value of a batched tensor inside vmap; return \
         it from the mapped function instead"

(* Constructs *)

(* [relanes m flags x] is [x], a value of [s], with the tensors at [flags] made
   lanes of [m]. *)
let relanes m s flags x =
  let leaves, _ = Nx.Ptree.flatten s x in
  Nx.Ptree.rebuild s ~like:x
    (List.map2
       (fun f (Nx.P x) -> if f then Nx.P (lane m x) else Nx.P x)
       flags leaves)

let lanes_of m s x =
  List.map (fun (Nx.P x) -> owns m x) (fst (Nx.Ptree.flatten s x))

let physicals m s x = Nx.Ptree.map s (fun _ x -> physical m x) x
let all_batched m s x = Nx.Ptree.map s (fun _ x -> batched m x) x

let rec answer : type r. t -> r Construct.t -> (unit -> r) option =
 fun m c ->
  match[@warning "@4@8"] c with
  | Lanes (axis, x) when named m axis ->
      Some
        (fun () ->
          if owns m x then physical m x
          else Nx.broadcast_to (Array.append [| m.size |] (Nx.shape x)) x)
  | Lanes (axis, x) ->
      if owns m x then
        Some
          (fun () ->
            let gathered = Construct.perform (Lanes (axis, physical m x)) in
            lane m (Nx.swapaxes 0 1 gathered))
      else None
  | Lane_index axis ->
      let ours =
        match axis with Some a -> named m a | None -> Option.is_none m.axis
      in
      if ours then Some (fun () -> lane m (Nx.arange Nx.int32 0 m.size 1))
      else None
  | Lane_count axis -> if named m axis then Some (fun () -> m.size) else None
  | Add (t, v) -> Some (fun () -> Construct.perform (Add (t, sum_lanes m v)))
  | Detach x ->
      if owns m x then
        Some (fun () -> lane m (Construct.perform (Detach (physical m x))))
      else None
  | Custom r -> Some (fun () -> custom m r)
  | Remat { p; q; f; args; recomputed } ->
      Some
        (fun () ->
          let flags = lanes_of m p args and out = ref [] in
          let f args =
            let y = install m (fun () -> f (relanes m p flags args)) in
            out := lanes_of m q y;
            physicals m q y
          in
          let y =
            Construct.perform
              (Remat { p; q; f; args = physicals m p args; recomputed })
          in
          relanes m q !out y)
  | Scan _ | Barrier _ -> None

(* A custom call passes on as the call of its rule batched: the rule runs under
   the map reinstalled, so the lanes it receives and captures are the map's
   again, and returns its results batched; the gradient of an argument the lanes
   share is summed over them. *)
and custom : type q. t -> q Construct.rule -> q =
 fun m r ->
  match r with
  | Jvp_rule { p; q; rule; args; value } ->
      let flags = lanes_of m p args in
      let rule args =
        let y, map = install m (fun () -> rule (relanes m p flags args)) in
        let map dargs =
          all_batched m q (install m (fun () -> map (relanes m p flags dargs)))
        in
        (all_batched m q y, map)
      in
      let value = Option.map (all_batched m q) value in
      let y =
        Construct.perform
          (Custom (Jvp_rule { p; q; rule; args = physicals m p args; value }))
      in
      Nx.Ptree.map q (fun _ y -> lane m y) y
  | Vjp_rule { p; q; rule; args } ->
      let flags = lanes_of m p args in
      let rule pargs =
        let y, pullback =
          install m (fun () -> rule (relanes m p flags pargs))
        in
        let pullback cts =
          let g =
            install m (fun () ->
                pullback (Nx.Ptree.map q (fun _ c -> lane m c) cts))
          in
          Structure.map2 m.entry p ~this:"the arguments"
            ~that:"the pullback's result"
            (fun _ x g -> if owns m x then batched m g else sum_lanes m g)
            args g
        in
        (all_batched m q y, pullback)
      in
      let y =
        Construct.perform
          (Custom (Vjp_rule { p; q; rule; args = physicals m p args }))
      in
      Nx.Ptree.map q (fun _ y -> lane m y) y

and install : type a. t -> (unit -> a) -> a =
 fun m f ->
  let owner = { Construct.owns = (fun x -> owns m x) } in
  let claims op = Construct.claims owner op in
  Construct.install
    {
      op = Some { run = (fun op -> run m op); claims };
      call = (fun c -> answer m c);
    }
    f
