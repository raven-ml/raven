(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let prefix = "state"

(* The run's description as its bytes. *)
let bytes s =
  Nx.create Nx.uint8
    [| String.length s |]
    (Array.init (String.length s) (fun i -> Char.code s.[i]))

let string x = String.init (Nx.numel x) (fun i -> Char.chr (Nx.item [ i ] x))

(* The state is saved from the host, where the key and the position lie: a save
   copies a value onto the host anyway. *)
let save (type v s d) file (module U : Nx.Ptree.S) (x : ((v, s, d) Nx.t, d) U.t)
    ~key ~position ~run =
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
              ("run", Nx.P (bytes run));
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
  let run =
    match Nx_io.Archive.find "run" a with
    | Some (Nx.P r) -> string (Nx.unpack Nx.uint8 (Nx.P r))
    | None -> failwith ("Tessera.Checkpoint.load: " ^ file ^ ": no run")
  in
  (x, key, Int64.to_int position, run)

let part (type v s d) file (module U : Nx.Ptree.S) ~path
    ~(like : ((v, s, d) Nx.t, d) U.t) =
  Nx_io.Archive.to_value
    ~prefix:(prefix ^ "." ^ path)
    (Nx.Ptree.instantiate (module U))
    ~like
    (Nx_io.load_safetensors file)
