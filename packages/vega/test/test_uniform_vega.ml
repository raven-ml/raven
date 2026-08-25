(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Uniform → Ptree.S bridge: structural optimiser step over Ptree.Make. Uses
   hand-crafted gradients to avoid a rune dependency in the vega test suite. *)

open Windtrap

(* ——— helpers ——— *)

let as_f32 (type a b) (x : (a, b) Nx.t) : Nx.float32_t =
  match Nx_core.Dtype.equal_witness (Nx.dtype x) Nx.float32 with
  | Some Type.Equal -> x
  | None -> failwith "expected a float32 leaf"

let vec32 xs = Nx.create Nx.float32 [| Array.length xs |] xs
let pack x = Nx.Ptree.P x

let check_t name shape values (t : Nx.float32_t) =
  let actual_shape = Nx.shape t in
  equal ~msg:(name ^ " shape") int (Array.length shape)
    (Array.length actual_shape);
  Array.iteri
    (fun i s ->
      equal ~msg:(Printf.sprintf "%s shape[%d]" name i) int s actual_shape.(i))
    shape;
  let actual = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t)) in
  equal ~msg:(name ^ " length") int (Array.length values) (Array.length actual);
  Array.iteri
    (fun i x ->
      equal ~msg:(Printf.sprintf "%s[%d]" name i) (float 1e-5) x actual.(i))
    values

(* ——— hand-written uniform structure ——— *)

module U = struct
  type 'a t = { w : 'a; b : 'a }

  let map (f : 'a -> 'b) { w; b } = { w = f w; b = f b }
  let map2 (f : 'a -> 'b -> 'c) a b = { w = f a.w b.w; b = f a.b b.b }

  let iter (f : 'a -> unit) { w; b } =
    f w;
    f b

  let fold (f : string -> 'acc -> 'a -> 'acc) acc { w; b } =
    f "b" (f "w" acc w) b

  let fold2 (f : string -> 'acc -> 'a -> 'b -> 'acc) acc a b =
    f "b" (f "w" acc a.w b.w) a.b b.b

  let names _ = { w = "w"; b = "b" }
end

module T = Nx.Ptree.Make (U)

(* Initial parameters and synthetic gradients. *)
let params =
  { U.w = pack (vec32 [| 1.0; -2.0; 3.0 |]); b = pack (vec32 [| 0.5 |]) }

let grads =
  { U.w = pack (vec32 [| 0.1; 0.2; 0.3 |]); b = pack (vec32 [| 0.05 |]) }

let test_adam_step_on_uniform () =
  let st = Vega.adam_init (module T) params in
  let p', _ = Vega.adam_step (module T) ~lr:0.1 st ~params ~grads in
  let { U.w = Nx.Ptree.P w'; b = Nx.Ptree.P b' } = p' in
  let w' = as_f32 w' and b' = as_f32 b' in
  check_t "w after adam step" [| 3 |] [| 0.9; -2.1; 2.9 |] w';
  check_t "b after adam step" [| 1 |] [| 0.4 |] b'

let test_clip_by_global_norm_on_uniform () =
  let g_clipped = Vega.clip_by_global_norm (module T) ~max_norm:10.0 grads in
  let { U.w = Nx.Ptree.P wc; _ } = g_clipped in
  (* Our gradients are small enough that clipping is a no-op. *)
  check_t "w after clip" [| 3 |] [| 0.1; 0.2; 0.3 |] (as_f32 wc)

let tests =
  [
    group "Vega over Ptree.Make"
      [
        test "adam_step preserves structure and applies update"
          test_adam_step_on_uniform;
        test "clip_by_global_norm works on uniform-backed gradients"
          test_clip_by_global_norm_on_uniform;
      ];
  ]

let () = run "vega uniform" tests
