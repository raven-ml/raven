(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Custom differentiation rules. Provides combinators for overriding automatic
   differentiation with user-supplied forward/backward (VJP) or forward/tangent
   (JVP) rules. *)

open Nx_core
module T = Nx

let custom_vjp (type a b c d res) ~(fwd : (a, b) T.t -> (c, d) T.t * res)
    ~(bwd : res -> (c, d) T.t -> (a, b) T.t) (x : (a, b) T.t) : (c, d) T.t =
  match Autodiff.query_ad_mode () with
  | Some `JVP | None -> fst (fwd x)
  | Some `VJP ->
      let residuals = ref None in
      let y_ref = ref (Obj.repr ()) in
      let cv_fwd () =
        let y, r = fwd x in
        residuals := Some r;
        y_ref := Obj.repr y;
        Obj.repr y
      in
      let cv_bwd get_grad acc_grad =
        let g : (c, d) T.t = Obj.obj (get_grad !y_ref) in
        (* residuals is guaranteed Some here: cv_fwd runs to completion before
           the VJP handler calls cv_bwd on the backward pass *)
        let dx = bwd (Option.get !residuals) g in
        acc_grad (Obj.repr x) (Obj.repr dx)
      in
      Obj.obj (Effect.perform (Autodiff.E_custom_vjp { cv_fwd; cv_bwd }))

let custom_vjps (type a b c d res) ~(fwd : (a, b) T.t list -> (c, d) T.t * res)
    ~(bwd : res -> (c, d) T.t -> (a, b) T.t list) (xs : (a, b) T.t list) :
    (c, d) T.t =
  match Autodiff.query_ad_mode () with
  | Some `JVP | None -> fst (fwd xs)
  | Some `VJP ->
      let residuals = ref None in
      let y_ref = ref (Obj.repr ()) in
      let cv_fwd () =
        let y, r = fwd xs in
        residuals := Some r;
        y_ref := Obj.repr y;
        Obj.repr y
      in
      let cv_bwd get_grad acc_grad =
        let g : (c, d) T.t = Obj.obj (get_grad !y_ref) in
        (* residuals is guaranteed Some here: cv_fwd runs to completion before
           the VJP handler calls cv_bwd on the backward pass *)
        let dxs = bwd (Option.get !residuals) g in
        List.iter2 (fun x dx -> acc_grad (Obj.repr x) (Obj.repr dx)) xs dxs
      in
      Obj.obj (Effect.perform (Autodiff.E_custom_vjp { cv_fwd; cv_bwd }))

let custom_jvp (type a b c d) ~(fwd : (a, b) T.t -> (c, d) T.t)
    ~(jvp_rule : (a, b) T.t -> (a, b) T.t -> (c, d) T.t * (c, d) T.t)
    (x : (a, b) T.t) : (c, d) T.t =
  match Autodiff.query_ad_mode () with
  | Some `VJP | None -> fwd x
  | Some `JVP ->
      let cj_jvp get_tangent =
        let tangent : (a, b) T.t = Obj.obj (get_tangent (Obj.repr x)) in
        let y, t = jvp_rule x tangent in
        (Obj.repr y, Obj.repr t)
      in
      Obj.obj (Effect.perform (Autodiff.E_custom_jvp { cj_jvp }))

let custom_jvps (type a b c d) ~(fwd : (a, b) T.t list -> (c, d) T.t)
    ~(jvp_rule : (a, b) T.t list -> (a, b) T.t list -> (c, d) T.t * (c, d) T.t)
    (xs : (a, b) T.t list) : (c, d) T.t =
  match Autodiff.query_ad_mode () with
  | Some `VJP | None -> fwd xs
  | Some `JVP ->
      let cj_jvp get_tangent =
        let tangents =
          List.map
            (fun x -> (Obj.obj (get_tangent (Obj.repr x)) : (a, b) T.t))
            xs
        in
        let y, t = jvp_rule xs tangents in
        (Obj.repr y, Obj.repr t)
      in
      Obj.obj (Effect.perform (Autodiff.E_custom_jvp { cj_jvp }))
