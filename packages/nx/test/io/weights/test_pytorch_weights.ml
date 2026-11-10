(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A real checkpoint loads as torch.load reads it: every tensor of its state
   dict has torch's dtype, shape and bytes, and its hyperparameters are
   Python's. The dune file says how to run it. *)

open Windtrap
module Pt = Nx_io.Pytorch

let env k =
  match Sys.getenv_opt k with
  | Some v -> v
  | None -> failwith (k ^ " is not set: the dune file says how to run this")

let field k = function
  | Pt.Dict kv -> List.assoc (Pt.String k) kv
  | _ -> failwith "no dict"

(* A value's dtype name, shape and bytes, as the reference stores them. *)
let stored (Nx.P x) =
  let k = Nx.Dtype.bits (Nx.dtype x) / 8 in
  let bytes =
    Nx.to_array (Nx.reshape [| -1 |] (Nx.bitcast Nx.uint8 (Nx.copy x)))
  in
  (Nx.Dtype.name (Nx.dtype x), Array.append (Nx.shape x) [| k |], bytes)

let reference ref name =
  let entry n = Option.get (Nx_io.Archive.find n ref) in
  let (Nx.P dt) = entry (name ^ ".dtype") in
  let dtype =
    String.init (Nx.numel dt) (fun i ->
        Char.chr (Nx.to_array (Nx.unpack Nx.uint8 (Nx.P dt))).(i))
  in
  let (Nx.P b) = entry name in
  let b = Nx.unpack Nx.uint8 (Nx.P b) in
  ( (* "torch.float32" is nx's "float32". *)
    String.sub dtype 6 (String.length dtype - 6),
    Nx.shape b,
    Nx.to_array (Nx.reshape [| -1 |] b) )

let args : Pt.value =
  let s k = Pt.String k in
  Dict
    [
      (s "repr_dim", Int 128);
      (s "latent_dim", Int 110);
      (s "num_layers", Int 4);
      (s "nhead", Int 4);
      (s "dim_feedforward", Int 1792);
      (s "max_seq_len", Int 256);
      (s "matryoshka_dims", s "16,32,64,128");
      (s "student_type", s "transformer");
      (s "enable_qk_norm", Bool false);
    ]

let () =
  let ckpt = Nx_io.load_pytorch (env "NX_IO_PT_CKPT") in
  let ref = Nx_io.load_npz (env "NX_IO_PT_REF") in
  let ours = Pt.tensors ckpt in
  let names =
    List.filter
      (fun n -> not (String.ends_with ~suffix:".dtype" n))
      (Nx_io.Archive.names ref)
  in
  let entry = triple string (array int) (array int) in
  exit
    (run "nx.io PyTorch weights"
       [
         test "its tensors are torch's, named by their state dict's keys"
           (fun () ->
             equal (list string) names (Nx_io.Archive.names ours);
             List.iter
               (fun n ->
                 equal ~msg:n entry (reference ref n)
                   (stored (Option.get (Nx_io.Archive.find n ours))))
               names);
         test "its hyperparameters are Python's" (fun () ->
             let scalars = function
               | Pt.Dict kv ->
                   List.map
                     (function
                       | Pt.String k, Pt.Int i -> (k, string_of_int i)
                       | Pt.String k, Pt.String s -> (k, s)
                       | Pt.String k, Pt.Bool b -> (k, string_of_bool b)
                       | _ -> ("?", "?"))
                     kv
               | _ -> []
             in
             equal
               (list (pair string string))
               (scalars args)
               (scalars (field "args" ckpt)));
       ])
