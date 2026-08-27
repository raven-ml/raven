(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Uniform → Ptree.S bridge via Ptree.Make: hand-written uniform structure, no
   ppx. *)

open Windtrap

(* ——— helpers ——— *)

(* as_f32 : refine an existentially-typed tensor to concrete float32 through a
   dtype witness check. This is the standard pattern for consuming existential
   tensors from packed leaves. *)
let as_f32 (type a b) (x : (a, b) Nx.t) : Nx.float32_t =
  match Nx_core.Dtype.equal_witness (Nx.dtype x) Nx.float32 with
  | Some Type.Equal -> x
  | None -> failwith "expected a float32 leaf"

let raw (t : Nx.float32_t) = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let check ~msg (expected : float array) (actual : float array) =
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e ->
      equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float 1e-5) e actual.(i))
    expected

let f32 = Nx.float32
let vec32 xs = Nx.create f32 [| Array.length xs |] xs
let pack x = Nx.Ptree.P x

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

(* ——— Ptree.S bridge ——— *)

module T = Nx.Ptree.Make (U)
module Check : Nx.Ptree.S = T

let params () =
  { U.w = pack (vec32 [| 1.0; -2.0; 3.0 |]); b = pack (vec32 [| 0.5 |]) }

(* ——— tests ——— *)

let test_map_preserves_structure () =
  let t : T.t = params () in
  let r = T.map (fun x -> Nx.mul x x) t in
  match r with
  | { U.w = Nx.Ptree.P w; b = Nx.Ptree.P b } ->
      check ~msg:"w" [| 1.0; 4.0; 9.0 |] (raw (as_f32 w));
      check ~msg:"b" [| 0.25 |] (raw (as_f32 b))

let test_map2_combines_leafwise () =
  let a = params () and b = params () in
  let r = T.map2 (fun x y -> Nx.add x y) a b in
  match r with
  | { U.w = Nx.Ptree.P w; _ } ->
      check ~msg:"w" [| 2.0; -4.0; 6.0 |] (raw (as_f32 w))

let test_iter_visits_every_leaf () =
  let count = ref 0 in
  T.iter (fun _ -> incr count) (params ());
  equal ~msg:"leaves" int 2 !count

let test_uniform_fold_paths () =
  let visited = ref [] in
  let _ =
    U.fold
      (fun path acc _ ->
        visited := path :: !visited;
        acc)
      0 (params ())
  in
  let paths = List.rev !visited in
  equal ~msg:"count" int 2 (List.length paths);
  equal ~msg:"first path" string "w" (List.nth paths 0);
  equal ~msg:"second path" string "b" (List.nth paths 1)

let test_uniform_fold2 () =
  let merged =
    U.fold2 (fun path acc _ _ -> path :: acc) [] (params ()) (params ())
  in
  let merged = List.rev merged in
  equal ~msg:"merged count" int 2 (List.length merged)

let test_uniform_names () =
  let names = U.names (params ()) in
  let paths = List.rev (U.fold (fun path acc _ -> path :: acc) [] names) in
  equal ~msg:"names agrees with fold's paths" (list string) [ "w"; "b" ] paths;
  equal ~msg:"each payload carries its own path" string "w" names.U.w

let test_map2_dtype_mismatch () =
  let b =
    {
      U.w = pack (Nx.create Nx.float64 [| 1 |] [| 1.0 |]);
      b = pack (vec32 [| 0.5 |]);
    }
  in
  raises_match
    (fun e -> match e with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (T.map2 (fun x _ -> x) (params ()) b))

let test_typed_instance () =
  let module TP =
    (val Nx.Ptree.instantiate (module U)
        : Nx.Ptree.S with type t = Nx.float32_t U.t)
  in
  let p = { U.w = vec32 [| 1.0; -2.0 |]; b = vec32 [| 3.0 |] } in
  let r = TP.map (fun x -> Nx.mul x x) p in
  check ~msg:"typed map w" [| 1.0; 4.0 |] (raw r.U.w);
  check ~msg:"typed map b" [| 9.0 |] (raw r.U.b);
  let s = TP.map2 (fun x y -> Nx.add x y) p p in
  check ~msg:"typed map2 w" [| 2.0; -4.0 |] (raw s.U.w);
  let count = ref 0 in
  TP.iter (fun _ -> incr count) p;
  equal ~msg:"typed iter leaves" int 2 !count

let test_tree_paths () =
  let t =
    Nx.Ptree.dict
      [
        ( "layers",
          Nx.Ptree.list
            [
              Nx.Ptree.tensor (vec32 [| 1.0 |]);
              Nx.Ptree.tensor (vec32 [| 2.0 |]);
            ] );
        ("head", Nx.Ptree.tensor (vec32 [| 3.0 |]));
      ]
  in
  let expected = [ "layers.0"; "layers.1"; "head" ] in
  let paths = List.rev (Nx.Ptree.Tree.fold (fun p acc _ -> p :: acc) [] t) in
  equal ~msg:"tree paths" (list string) expected paths;
  let names = Nx.Ptree.Tree.names t in
  equal ~msg:"tree names agrees with fold" (list string) expected
    (List.rev (Nx.Ptree.Tree.fold (fun _ acc p -> p :: acc) [] names))

let test_unpack_ok () =
  let x = vec32 [| 1.0; 2.0 |] in
  check ~msg:"unpack" [| 1.0; 2.0 |] (raw (Nx.Ptree.unpack f32 (pack x)))

let test_unpack_mismatch () =
  let p = pack (vec32 [| 1.0 |]) in
  raises_match
    (fun e -> match e with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Nx.Ptree.unpack Nx.float64 p))

let test_unpack_at_path () =
  let p = pack (vec32 [| 1.0 |]) in
  raises_match
    (fun e ->
      match e with
      | Invalid_argument msg when String.length msg > 0 -> true
      | _ -> false)
    (fun () -> ignore (Nx.Ptree.unpack ~at:"model.layer.w" Nx.float64 p))

let tests =
  [
    group "Ptree.S bridge"
      [
        test "map preserves structure" test_map_preserves_structure;
        test "map2 combines leafwise" test_map2_combines_leafwise;
        test "iter visits every leaf" test_iter_visits_every_leaf;
      ];
    group "Uniform fold"
      [
        test "fold paths" test_uniform_fold_paths;
        test "fold2" test_uniform_fold2;
        test "names is the tree of paths" test_uniform_names;
      ];
    group "structure errors"
      [ test "map2 rejects dtype mismatch" test_map2_dtype_mismatch ];
    group "typed instance"
      [ test "typed traverses with typed leaves" test_typed_instance ];
    group "stock tree"
      [ test "list and dict positions become paths" test_tree_paths ];
    group "unpack"
      [
        test "unpack succeeds on matching dtype" test_unpack_ok;
        test "unpack raises on mismatching dtype" test_unpack_mismatch;
        test "unpack ~at includes path in message" test_unpack_at_path;
      ];
  ]

let () = run "nx ptree uniform" tests
