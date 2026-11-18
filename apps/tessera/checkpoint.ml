(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let prefix = "state"

(* The state is saved from the host, where the key and the position lie: a save
   copies a value onto the host anyway. *)
let save (type v s d) file (module U : Nx.Ptree.S) (x : ((v, s, d) Nx.t, d) U.t)
    ~key ~position =
  let host = Nx.Ptree.place (module U) Nx.Host.on x in
  let a =
    Nx_io.Archive.(
      union
        [
          of_value ~prefix (Nx.Ptree.instantiate (module U)) host;
          of_list
            [
              ("key", Nx.P (Nx.Rng.to_tensor key));
              ("position", Nx.P (Nx.scalar Nx.int64 (Int64.of_int position)));
            ];
        ])
  in
  Nx_io.save_safetensors file a

let load (type v s d) file (module U : Nx.Ptree.S)
    ~(like : ((v, s, d) Nx.t, d) U.t) =
  let a = Nx_io.load_safetensors file in
  let x =
    Nx_io.Archive.to_value ~prefix (Nx.Ptree.instantiate (module U)) ~like a
  in
  let key =
    Nx.Rng.of_tensor
      (Nx.copy (Nx_io.Archive.tensor ~shape:[| 2 |] Nx.int32 "key" a))
  in
  let position =
    Nx.item [] (Nx_io.Archive.tensor ~shape:[||] Nx.int64 "position" a)
  in
  (x, key, Int64.to_int position)

let fresh s x =
  let copy = Nx.Ptree.map s (fun _ a -> Nx.copy a) in
  Rune.jit Nx.Ptree.(s @-> returns s) copy x
