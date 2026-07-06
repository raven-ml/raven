(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  tracked : Tensor_map.Ids.t;
  mutable cotangents : Tensor_map.t;
  pulls : (unit -> unit) Dynarray.t;
}

let create () =
  {
    tracked = Tensor_map.Ids.create ();
    cotangents = Tensor_map.create ();
    pulls = Dynarray.create ();
  }

let track tape x = Tensor_map.Ids.add tape.tracked x
let tracked tape x = Tensor_map.Ids.mem tape.tracked x
let record tape pull = Dynarray.add_last tape.pulls pull

let backward tape =
  for i = Dynarray.length tape.pulls - 1 downto 0 do
    (Dynarray.get tape.pulls i) ()
  done

let find tape x = Tensor_map.find tape.cotangents x

let accumulate tape x g =
  (* Materialize the stored cotangent: contributions can be lazy views
     (broadcasts, transposes), and later pulls may reshape them. *)
  let g =
    match find tape x with None -> Nx.contiguous g | Some acc -> Nx.add acc g
  in
  Tensor_map.set tape.cotangents x g

let cotangent tape x =
  match find tape x with Some g -> g | None -> Nx.zeros_like x

let reset_cotangents tape = tape.cotangents <- Tensor_map.create ()
