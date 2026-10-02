(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type axis = unit Type.Id.t
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
  | Scan : Scan.request -> Scan.result t
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
  | Reduce (_, _, x) -> owns x
  | Move (x, _) -> owns x
  | Matmul (a, b) -> owns a || owns b
  | Compare _ | Where _ | Scan _ | Arg_reduce _ | Sort _ | Argsort _ | Pad _
  | Cat _ | Convert _ | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _
  | Fold _ | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _
  | Svd _ | Eig _ | Eigh _ | Solve_triangular _ | Place _ | Read _ | Fma _
  | Check _ ->
      List.exists (fun (Nx.P x) -> owns x) (Nx.Op.operands op)

type _ Effect.t += Construct : 'r t -> 'r Effect.t

let default : type r. r t -> r = function
  | Scan _ -> raise Scan.Not_staged
  | Compiled { p; q; f; args; compiler } -> compiler.run p q f args
  | Remat { f; args; _ } -> f args
  | Barrier { values; _ } -> values
  | Custom (Jvp_rule { value = Some y; _ }) -> y
  | Custom (Jvp_rule { rule; args; value = None; _ }) -> fst (rule args)
  | Custom (Vjp_rule { rule; args; _ }) -> fst (rule args)
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

let scan r =
  match perform (Scan r) with
  | r -> r
  | exception Scan.Not_staged -> Scan.fold r

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
