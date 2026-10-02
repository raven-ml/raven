(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Ptree = Nx.Ptree
module Ops = Tolk.Ops
module Repr = Nx.Repr

(* Values *)

let one x = [ Nx.P x ]

(* [results op r] is the tensors of [op]'s result [r]. *)
let results : type r. r Nx.Op.t -> r -> Nx.packed list =
 fun op r ->
  match[@warning "@4@8"] op with
  | Unary _ -> one r
  | Binary _ -> one r
  | Compare _ -> one r
  | Where _ -> one r
  | Fma _ -> one r
  | Reduce _ -> one r
  | Scan _ -> one r
  | Arg_reduce _ -> one r
  | Sort _ -> one r
  | Argsort _ -> one r
  | Pad _ -> one r
  | Cat _ -> one r
  | Convert _ -> one r
  | Threefry _ -> one r
  | Gather _ -> one r
  | Scatter _ -> one r
  | Update _ -> one r
  | Unfold _ -> one r
  | Fold _ -> one r
  | Matmul _ -> one r
  | Fft _ -> one r
  | Rfft _ -> one r
  | Irfft _ -> one r
  | Contiguous _ -> one r
  | Cholesky _ -> one r
  | Solve_triangular _ -> one r
  | Move _ -> one r
  | Place _ -> one r
  | Qr _ ->
      let a, b = r in
      [ Nx.P a; Nx.P b ]
  | Lu _ ->
      let a, b, c = r in
      [ Nx.P a; Nx.P b; Nx.P c ]
  | Svd _ ->
      let a, b, c = r in
      [ Nx.P a; Nx.P b; Nx.P c ]
  | Eig _ ->
      let a, b = r in
      Nx.P a :: Option.to_list (Option.map (fun b -> Nx.P b) b)
  | Eigh _ ->
      let a, b = r in
      Nx.P a :: Option.to_list (Option.map (fun b -> Nx.P b) b)
  | Read _ | Check _ -> []

(* [made c r] is the tensors of the construct [c]'s answer [r]. *)
let made : type r. r Construct.t -> r -> Nx.packed list =
 fun c r ->
  match[@warning "@4@8"] c with
  | Scan _ -> r.Scan.r_carry @ r.r_ys
  | Compiled { q; _ } -> fst (Ptree.flatten q r)
  | Remat { q; _ } -> fst (Ptree.flatten q r)
  | Barrier _ -> r
  | Custom (Jvp_rule { q; _ }) -> fst (Ptree.flatten q r)
  | Custom (Vjp_rule { q; _ }) -> fst (Ptree.flatten q r)
  | Lanes _ -> one r
  | Lane_index _ -> one r
  | Detach _ -> one r
  | Lane_count _ | Add _ -> []

(* Substitutions *)

(* [substituting owner s f] is [f ()] with each value [owner] owns that an
   operation or a construct of its extent reads replaced by [s]'s for it, in the
   callbacks of the constructs too. *)
let substituting (owner : Construct.owner) (s : mapper) f =
  let leaf (Nx.P x) = Nx.P (s.f x) in
  let leaves = List.map leaf in
  let rec install : 'a. (unit -> 'a) -> 'a =
   fun f ->
    let claims op = Construct.claims owner op in
    Construct.install
      { op = Some { run = (fun op -> eval (map_operands s op)); claims }; call }
      f
  and call : type r. r Construct.t -> (unit -> r) option =
   fun c ->
    let again c = Some (fun () -> Construct.perform c) in
    let args p a = Ptree.map p (fun _ x -> s.f x) a in
    match[@warning "@4@8"] c with
    | Scan r ->
        again
          (Scan
             {
               r with
               req_carry = leaves r.req_carry;
               req_xs = leaves r.req_xs;
               req_step = (fun c x -> install (fun () -> r.req_step c x));
             })
    | Compiled k ->
        again
          (Compiled
             {
               k with
               args = args k.p k.args;
               f = (fun a -> install (fun () -> k.f a));
             })
    | Remat k ->
        again
          (Remat
             {
               k with
               args = args k.p k.args;
               f = (fun a -> install (fun () -> k.f a));
             })
    | Barrier { values; after } ->
        again (Barrier { values = leaves values; after = leaves after })
    | Custom (Jvp_rule k) ->
        let rule a =
          let y, map = install (fun () -> k.rule a) in
          (y, fun da -> install (fun () -> map da))
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
          let y, pullback = install (fun () -> k.rule a) in
          (y, fun ct -> install (fun () -> pullback ct))
        in
        again (Custom (Vjp_rule { k with args = args k.p k.args; rule }))
    | Lanes (axis, x) -> again (Lanes (axis, s.f x))
    | Detach x -> again (Detach (s.f x))
    | Add (t, v) -> again (Add (t, s.f v))
    | Lane_index _ | Lane_count _ -> None
  in
  install f

(* Numbering *)

(* A value made by a movement or a placement, and how to make it again over the
   values a substitution gives for its operand. *)
type recipe = Recipe : ('a, 'b) Nx.t * (mapper -> ('a, 'b) Nx.t) -> recipe

let recipe : type r. r Nx.Op.t -> r -> recipe option =
 fun op r ->
  match[@warning "@4@8"] op with
  | Move (x, m) -> Some (Recipe (r, fun s -> eval (Move (s.f x, m))))
  | Place (p, x) -> Some (Recipe (r, fun s -> eval (Place (p, s.f x))))
  | Unary _ | Binary _ | Compare _ | Where _ | Fma _ | Reduce _ | Scan _
  | Arg_reduce _ | Sort _ | Argsort _ | Pad _ | Cat _ | Convert _ | Threefry _
  | Gather _ | Scatter _ | Update _ | Unfold _ | Fold _ | Matmul _ | Fft _
  | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _ | Svd _ | Eig _
  | Eigh _ | Solve_triangular _ | Read _ | Check _ ->
      None

let id x =
  match Repr.v x with
  | Traced t -> Some (Repr.Traced.id t)
  | Host _ | Placed _ -> None

(* A run's values in the order its operations and constructs made them, and the
   recipe of each it made by a movement or a placement, by identity. *)
type numbered = { values : Nx.packed array; recipes : (int, recipe) Hashtbl.t }

(* [numbering f] is [f ()] and the values the operations and the constructs of
   its extent made, each operation and construct passed on unchanged. A scan
   that no trace stages folds outside the extent, so that only its results are
   numbered: whether a trace stages a scan depends on more than the dtypes,
   shapes and placements of the function's arguments, such as the lanes of a map
   around the call. *)
let numbering f =
  let values = ref [] and recipes = Hashtbl.create 16 in
  let note l = values := List.rev_append l !values in
  let run : type r. r Nx.Op.t -> r =
   fun op ->
    let r = eval op in
    note (results op r);
    (match recipe op r with
    | Some (Recipe (y, _) as m) ->
        Option.iter (fun k -> Hashtbl.replace recipes k m) (id y)
    | None -> ());
    r
  in
  let call : type r. r Construct.t -> (unit -> r) option =
   fun c ->
    Some
      (fun () ->
        let r : r =
          match[@warning "@4@8"] c with
          | Scan q -> Construct.scan q
          | Compiled _ | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _
          | Lane_count _ | Add _ | Detach _ ->
              Construct.perform c
        in
        note (made c r);
        r)
  in
  let y =
    Construct.install { op = Some { run; claims = (fun _ -> true) }; call } f
  in
  (y, { values = Array.of_list (List.rev !values); recipes })

(* Traces *)

(* [fresh s p dt shape] is a value of [s] at [p] of [dt] and [shape] that no
   operation computes: a stand-in for a value from outside the trace. *)
let fresh s p dt shape = Lower.argument s ~slot:(Ops.unique_num ()) p dt shape
let same_axis (a : Construct.axis) b = Type.Id.uid a = Type.Id.uid b

(* [traced s counts f] is [f ()] traced in [s], every construct no trace stages
   answered by its default, traced in [s] too, but for a map's collectives: the
   lane count of the map around the call, which [counts] records, and a stand-in
   for a gathering or a lane's index, which the plan cannot compute as a run
   under the map does. *)
let rec traced :
    'a. Lower.scope -> (Construct.axis * int) list ref -> (unit -> 'a) -> 'a =
 fun s counts f ->
  let count axis =
    let n = Construct.perform (Lane_count axis) in
    if not (List.exists (fun (a, _) -> same_axis a axis) !counts) then
      counts := (axis, n) :: !counts;
    n
  in
  let call : type r. r Construct.t -> (unit -> r) option =
   fun c ->
    let default () =
      Some (fun () -> traced s counts (fun () -> Construct.default c))
    in
    match[@warning "@4@8"] c with
    | Lane_count axis -> Some (fun () -> count axis)
    | Add _ -> Some ignore
    | Lanes (axis, x) ->
        Some
          (fun () ->
            let shape = Array.append [| count axis |] (Nx.shape x) in
            fresh s (Nx.placement x) (Nx.dtype x) shape)
    | Lane_index _ -> Some (fun () -> fresh s Nx.Placement.host Nx.int32 [||])
    | Scan _ -> default ()
    | Compiled _ -> default ()
    | Remat _ -> default ()
    | Barrier _ -> default ()
    | Custom _ -> default ()
    | Detach _ -> default ()
  in
  Construct.install { op = None; call } (fun () -> Staged.install s f)

(* Plans *)

(* A residual: an argument, by its leaf's index, or a value the forward pass
   computes, by its number in a run. *)
type residual = Arg of int | Value of int

(* What a residual of the forward trace is in another trace. *)
type stand_in = { stand_in : 'a 'b. residual -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let jit_error fmt = Printf.ksprintf (fun m -> raise (Lower.Jit_error m)) fmt

let nondeterministic () =
  jit_error
    "Rune.jit: the function computed other values on this call than when its \
     reverse mode was split: a compiled function must be deterministic, \
     computing the same operations whenever its arguments have the same \
     dtypes, shapes and placements"

let same_value (Nx.P a) (Nx.P b) =
  Nx_dtype.equal (Nx.dtype a) (Nx.dtype b) && Nx.shape a = Nx.shape b

let plan (type p q) (p : p Ptree.t) (q : q Ptree.t) (vjp : (p, q) Construct.vjp)
    (a : p) : (Construct.axis * int) list * (p, q) Construct.split =
  let leaves, _ = Ptree.flatten p a in
  let standing s like =
    List.map
      (fun (Nx.P x) ->
        Nx.P (fresh s (Nx.placement x) (Nx.dtype x) (Nx.shape x)))
      like
  in
  let finally s f = Fun.protect ~finally:(fun () -> Lower.finish s) f in
  let counts = ref [] in
  (* The forward pass, traced keeping its tape. *)
  let fw = Lower.scope ~renderer:Tolk_engine.renderer in
  let params = standing fw leaves in
  let (y, transpose), numbered =
    finally fw @@ fun () ->
    traced fw counts (fun () ->
        numbering (fun () -> vjp (Ptree.rebuild p ~like:a params)))
  in
  (* The leaf of each argument, and the number of each value the forward trace
     computed, by identity. *)
  let index = Hashtbl.create 16 in
  List.iteri
    (fun j (Nx.P x) -> Hashtbl.replace index (Option.get (id x)) j)
    params;
  let first = Hashtbl.create 64 in
  Array.iteri
    (fun k (Nx.P v) ->
      match id v with
      | Some i when Lower.traces fw v && not (Hashtbl.mem first i) ->
          Hashtbl.add first i k
      | Some _ | None -> ())
    numbered.values;
  (* [subst stand_in] substitutes a value of the forward trace: an argument or a
     computed value by [stand_in]'s for its residual, a movement or a placement
     by itself made again over the substitution, and a constant by itself. *)
  let subst { stand_in } =
    let rec f : type a b. (a, b) Nx.t -> (a, b) Nx.t =
     fun v ->
      match id v with
      | Some i when Lower.traces fw v -> (
          match Hashtbl.find_opt index i with
          | Some j -> stand_in (Arg j) v
          | None -> (
              match Hashtbl.find_opt numbered.recipes i with
              | Some (Recipe (y, remake)) -> (
                  match Nx_dtype.equal_witness (Nx.dtype y) (Nx.dtype v) with
                  | Some Type.Equal -> remake { f }
                  | None -> assert false (* One value, one dtype. *))
              | None -> (
                  if Lower.is_constant v then v
                  else
                    match Hashtbl.find_opt first i with
                    | Some k -> stand_in (Value k) v
                    | None ->
                        jit_error
                          "Rune.jit: cannot split the reverse mode of the \
                           function: its transpose reads a value its forward \
                           pass made outside its operations and constructs")))
      | Some _ | None -> v
    in
    { f }
  in
  let owner = { Construct.owns = (fun x -> Lower.traces fw x) } in
  (* Its transpose, against dense cotangents, each residual it reads in the
     order it first reads them, stood for by a value of its own. *)
  let bw = Lower.scope ~renderer:Tolk_engine.renderer in
  let outputs =
    List.filter
      (fun (Nx.P y) -> Linear.differentiable y)
      (fst (Ptree.flatten q y))
  in
  let reached = ref [] and stand_ins = Hashtbl.create 16 in
  let discover (type a b) r (v : (a, b) Nx.t) : (a, b) Nx.t =
    match Hashtbl.find_opt stand_ins r with
    | Some w -> Nx.unpack (Nx.dtype v) w
    | None ->
        let w = fresh bw (Nx.placement v) (Nx.dtype v) (Nx.shape v) in
        reached := r :: !reached;
        Hashtbl.add stand_ins r (Nx.P w);
        w
  in
  ignore
    ( finally bw @@ fun () ->
      traced bw counts (fun () ->
          substituting owner
            (subst { stand_in = discover })
            (fun () -> transpose (standing bw outputs))) );
  let residuals = List.rev !reached in
  let forward a =
    let (y, _), n = numbering (fun () -> vjp a) in
    if Array.length n.values <> Array.length numbered.values then
      nondeterministic ();
    let computed k =
      if not (same_value numbered.values.(k) n.values.(k)) then
        nondeterministic ();
      n.values.(k)
    in
    ( y,
      List.filter_map
        (function Value k -> Some (computed k) | Arg _ -> None)
        residuals )
  in
  let residuals_of a computed =
    let leaves, _ = Ptree.flatten p a in
    let rec go rs computed =
      match (rs, computed) with
      | Arg i :: rs, computed -> List.nth leaves i :: go rs computed
      | Value _ :: rs, c :: computed -> c :: go rs computed
      | [], [] -> []
      | Value _ :: _, [] | [], _ :: _ ->
          assert false (* A computed residual per [Value]. *)
    in
    go residuals computed
  in
  let position = Hashtbl.create 16 in
  List.iteri (fun k r -> Hashtbl.replace position r k) residuals;
  let backward (res, cts) =
    let res = Array.of_list res in
    let read r v = Nx.unpack (Nx.dtype v) res.(Hashtbl.find position r) in
    Total.discarding (fun () ->
        substituting owner (subst { stand_in = read }) (fun () -> transpose cts))
  in
  (!counts, { forward; backward; residuals = residuals_of })
