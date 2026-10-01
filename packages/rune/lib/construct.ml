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

type _ t =
  | Scan : Scan.request -> Scan.result t
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
  | Svd _ | Eig _ | Eigh _ | Solve_triangular _ | Place _ | Read _ ->
      List.exists (fun (Nx.P x) -> owns x) (Nx.Op.operands op)

type _ Effect.t += Construct : 'r t -> 'r Effect.t

let default : type r. r t -> r = function
  | Scan _ -> raise Scan.Not_staged
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

let perform c =
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
  Effect.Deep.match_with f () { retc = Fun.id; exnc; effc }
