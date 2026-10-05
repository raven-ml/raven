(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type axis = unit Type.Id.t
type map = unit ref

let fresh_map () = ref ()

type ('a, 'b) total = ('a, 'b) Nx.t Type.Id.t

type 'q rule =
  | Jvp_rule : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      rule : 'p -> 'q * ('p -> 'q);
      args : 'p;
      value : 'q option;
    }
      -> 'q rule
  | Vjp_rule : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      rule : 'p -> 'q * ('q -> 'p);
      args : 'p;
    }
      -> 'q rule

type (_, _, _, _) step =
  | Forward : bool list -> ('p, 'q, 'p, 'q * Nx.packed list) step
  | Backward :
      bool list
      -> ('p, 'q, Nx.packed list * Nx.packed list, Nx.packed list) step
  | Jvp : bool list -> ('p, 'q, 'p * 'p, 'q * 'q) step
  | Vmap : {
      lanes : bool list;
      size : int;
      axis : axis option;
    }
      -> ('p, 'q, 'p, 'q) step
  | Totals :
      ('a, 'b) total
      -> ('p, 'q, 'p * ('a, 'b) Nx.t, 'q * ('a, 'b) Nx.t) step
  | Discarding : ('p, 'q, 'p, 'q) step

let same_axis a b =
  match (a, b) with
  | None, None -> true
  | Some a, Some b -> Type.Id.uid a = Type.Id.uid b
  | _ -> false

let same_step : type p q a b c d.
    (p, q, a, b) step -> (p, q, c, d) step -> (a * b, c * d) Type.eq option =
 fun s s' ->
  match (s, s') with
  | Forward f, Forward f' when f = f' -> Some Equal
  | Backward f, Backward f' when f = f' -> Some Equal
  | Jvp f, Jvp f' when f = f' -> Some Equal
  | Vmap v, Vmap v'
    when v.lanes = v'.lanes && v.size = v'.size && same_axis v.axis v'.axis ->
      Some Equal
  | Totals t, Totals t' -> (
      match Type.Id.provably_equal t t' with
      | Some Equal -> Some Equal
      | None -> None)
  | Discarding, Discarding -> Some Equal
  | ( (Forward _ | Backward _ | Jvp _ | Vmap _ | Totals _ | Discarding),
      (Forward _ | Backward _ | Jvp _ | Vmap _ | Totals _ | Discarding) ) ->
      None

type ('p, 'q) split = {
  forward : 'p -> 'q * Nx.packed list;
  backward : Nx.packed list * Nx.packed list -> Nx.packed list;
  residuals : 'p -> Nx.packed list -> Nx.packed list;
}

type ('p, 'q) vjp = 'p -> 'q * (Nx.packed list -> Nx.packed list)

type ('p, 'q) compiler = {
  run : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'q;
  derive : 'p2 'q2. ('p, 'q, 'p2, 'q2) step -> ('p2, 'q2) compiler;
  split :
    bool list ->
    'p Nx.Ptree.t ->
    'q Nx.Ptree.t ->
    ('p, 'q) vjp ->
    'p ->
    ('p, 'q) split;
}

module Packed = struct
  type _ t = Nx.packed list

  let walk c l =
    Nx.Ptree.Walk.list (fun c (Nx.P x) -> Nx.P (Nx.Ptree.Walk.tensor c x)) c l
end

let packed : Nx.packed list Nx.Ptree.t = Nx.Ptree.instantiate (module Packed)

type _ t =
  | Loop : Trips.request -> Trips.result t
  | Compiled : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      f : 'p -> 'q;
      args : 'p;
      compiler : ('p, 'q) compiler;
    }
      -> 'q t
  | Remat : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      f : 'p -> 'q;
      args : 'p;
      recomputed : bool;
    }
      -> 'q t
  | Barrier : {
      values : Nx.packed list;
      after : Nx.packed list;
    }
      -> Nx.packed list t
  | Custom : 'q rule -> 'q t
  | Root : {
      x : 'x Nx.Ptree.t;
      residual : 'x -> 'x;
      solve : unit -> 'x;
      linear_solve : ('x -> 'x) -> 'x -> 'x;
    }
      -> 'x t
  | At_map : {
      map : map;
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      f : 'p -> 'q;
      x : 'p;
    }
      -> 'q t
  | Lanes : axis * ('a, 'b) Nx.t -> ('a, 'b) Nx.t t
  | Lane_index : axis option -> (int32, Nx.int32_elt) Nx.t t
  | Lane_count : axis -> int t
  | Add : ('a, 'b) total * ('a, 'b) Nx.t -> unit t
  | Detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t t

type interpreter = {
  op : Nx.Op.interpreter option;
  call : 'r. 'r t -> (unit -> 'r) option;
}

type owner = { owns : 'a 'b. ('a, 'b) Nx.t -> bool }

let claims : type r. owner -> r Nx.Op.t -> bool =
 fun { owns } op ->
  match[@warning "@4@8"] op with
  | Unary (_, x) -> owns x
  | Binary (_, a, b) -> owns a || owns b
  | Compare (_, a, b) -> owns a || owns b
  | Where (c, a, b) -> owns c || owns a || owns b
  | Fma (a, b, c) -> owns a || owns b || owns c
  | Reduce (_, _, x) -> owns x
  | Scan (_, _, x) -> owns x
  | Arg_reduce (_, _, x) -> owns x
  | Sort { x; _ } -> owns x
  | Argsort { x; _ } -> owns x
  | Group { x; _ } -> owns x
  | Pad (_, _, x) -> owns x
  | Cat (_, xs) -> List.exists owns xs
  | Convert (_, _, x) -> owns x
  | Threefry (key, ctr) -> owns key || owns ctr
  | Gather (_, indices, x) -> owns x || owns indices
  | Scatter { indices; updates; into; _ } ->
      owns into || owns indices || owns updates
  | Update (x, starts, v) -> owns x || owns starts || owns v
  | Unfold { x; _ } -> owns x
  | Fold { x; _ } -> owns x
  | Matmul (a, b) -> owns a || owns b
  | Fft { x; _ } -> owns x
  | Rfft { x; _ } -> owns x
  | Irfft { x; _ } -> owns x
  | Contiguous x -> owns x
  | Cholesky { x; _ } -> owns x
  | Qr { x; _ } -> owns x
  | Lu x -> owns x
  | Svd { x; _ } -> owns x
  | Eig { x; _ } -> owns x
  | Eigh { x; _ } -> owns x
  | Solve_triangular { a; b; _ } -> owns a || owns b
  | Move (x, _) -> owns x
  | Place (_, x) -> owns x
  | Read { x; _ } -> owns x
  | Check { ok; _ } -> owns ok

type _ Effect.t += Construct : 'r t -> 'r Effect.t

let default : type r. r t -> r = function
  | Loop _ -> raise Trips.Not_staged
  | Compiled { p; q; f; args; compiler } -> compiler.run p q f args
  | Remat { f; args; _ } -> f args
  | Barrier { values; _ } -> values
  | Custom (Jvp_rule { value = Some y; _ }) -> y
  | Custom (Jvp_rule { rule; args; value = None; _ }) -> fst (rule args)
  | Custom (Vjp_rule { rule; args; _ }) -> fst (rule args)
  | Root { solve; _ } -> solve ()
  | At_map _ ->
      invalid_arg
        "Rune.root: linear_solve's operator was applied after linear_solve \
         returned, or inside a Rune.jit it called"
  | Lanes (_, x) -> Nx.unsqueeze ~axes:[ 0 ] x
  | Lane_index _ -> Nx.scalar Nx.int32 0l
  | Lane_count _ -> 1
  | Add _ -> ()
  | Detach x -> x

(* How many installations are live, on any domain. While none is, a construct is
   answered by its default without performing an effect, which would raise
   [Effect.Unhandled] on every compiled call outside a transformation. The count
   is global because a suspended fiber may resume on another domain. *)
let installed = Atomic.make 0

let perform c =
  if Atomic.get installed = 0 then default c
  else
    let e = Construct c in
    match Effect.perform e with
    | r -> r
    | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e -> default c

let loop r =
  match perform (Loop r) with
  | r -> r
  | exception Trips.Not_staged -> Trips.fold r

let resume answer k =
  match answer () with
  | r -> Effect.Deep.continue k r
  | exception e ->
      Effect.Deep.discontinue_with_backtrace k e (Printexc.get_raw_backtrace ())

let install i f =
  let effc : type c a.
      c Effect.t -> ((c, a) Effect.Deep.continuation -> a) option = function
    | Construct c -> (
        match i.call c with
        | None -> None
        | Some answer -> Some (resume answer)
        | exception e ->
            let bt = Printexc.get_raw_backtrace () in
            Some (fun k -> Effect.Deep.discontinue_with_backtrace k e bt))
    | _ -> None
  in
  let f =
    match i.op with None -> f | Some o -> fun () -> Nx.Op.intercept o f
  in
  let exnc e =
    Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())
  in
  Atomic.incr installed;
  Fun.protect ~finally:(fun () -> Atomic.decr installed) @@ fun () ->
  Effect.Deep.match_with f () { retc = Fun.id; exnc; effc }

let substituting owner (s : Nx.Op.mapper) f =
  let s : Nx.Op.mapper = { f = (fun x -> if owner.owns x then s.f x else x) } in
  let leaf (Nx.P x) = Nx.P (s.f x) in
  let leaves = List.map leaf in
  let rec reinstall : 'a. (unit -> 'a) -> 'a =
   fun f ->
    let claims op = claims owner op in
    let run op = Nx.Op.eval (Nx.Op.map_operands s op) in
    install { op = Some { run; claims }; call } f
  and call : type r. r t -> (unit -> r) option =
   fun c ->
    let again c = Some (fun () -> perform c) in
    let args p a = Nx.Ptree.map p (fun _ x -> s.f x) a in
    match[@warning "@4@8"] c with
    | Loop r ->
        let req_trips : Trips.trips =
          match r.req_trips with
          | Rows rows -> Rows { rows with xs = leaves rows.xs }
          | Until stop ->
              Until
                {
                  stop with
                  until = (fun c -> reinstall (fun () -> stop.until c));
                }
        in
        again
          (Loop
             {
               req_carry = leaves r.req_carry;
               req_trips;
               req_step = (fun c x -> reinstall (fun () -> r.req_step c x));
             })
    | Compiled k ->
        again
          (Compiled
             {
               k with
               args = args k.p k.args;
               f = (fun a -> reinstall (fun () -> k.f a));
             })
    | Remat k ->
        again
          (Remat
             {
               k with
               args = args k.p k.args;
               f = (fun a -> reinstall (fun () -> k.f a));
             })
    | Barrier { values; after } ->
        again (Barrier { values = leaves values; after = leaves after })
    | Custom (Jvp_rule k) ->
        let rule a =
          let y, map = reinstall (fun () -> k.rule a) in
          (y, fun da -> reinstall (fun () -> map da))
        in
        again
          (Custom
             (Jvp_rule
                {
                  k with
                  args = args k.p k.args;
                  value = Option.map (args k.q) k.value;
                  rule;
                }))
    | Custom (Vjp_rule k) ->
        let rule a =
          let y, pullback = reinstall (fun () -> k.rule a) in
          (y, fun ct -> reinstall (fun () -> pullback ct))
        in
        again (Custom (Vjp_rule { k with args = args k.p k.args; rule }))
    | Root k ->
        let residual x = reinstall (fun () -> k.residual x)
        and solve () = reinstall k.solve
        and linear_solve op b = reinstall (fun () -> k.linear_solve op b) in
        again (Root { k with residual; solve; linear_solve })
    | At_map k ->
        again
          (At_map
             {
               k with
               x = args k.p k.x;
               f = (fun a -> reinstall (fun () -> k.f a));
             })
    | Lanes (axis, x) -> again (Lanes (axis, s.f x))
    | Detach x -> again (Detach (s.f x))
    | Add (t, v) -> again (Add (t, s.f v))
    | Lane_index _ | Lane_count _ -> None
  in
  reinstall f
