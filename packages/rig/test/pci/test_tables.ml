(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig.pci's tables against a list of bindings, the newest first. *)

open Windtrap
module T = Rig_pci.Tables.Address

(* Keys few enough that most of them are bound several times, and changes
   enough to grow a table made for one binding several times. *)
let keys = 16
let changes = 300

(* A change: [0] adds, [1] replaces, [2] removes, a key's binding. *)
let change = Gen.triple (Gen.int_range 0 2) (Gen.int_range 0 (keys - 1)) Gen.int

let rec replaced k v = function
  | [] -> [ (k, v) ]
  | (k', _) :: rest when k' = k -> (k, v) :: rest
  | b :: rest -> b :: replaced k v rest

let apply (t, model) (op, k, v) =
  match op with
  | 0 ->
      T.add t k v;
      (t, (k, v) :: model)
  | 1 ->
      T.replace t k v;
      (t, if List.mem_assoc k model then replaced k v model else (k, v) :: model)
  | _ ->
      T.remove t k;
      (t, List.remove_assoc k model)

(* After each change, each key's binding is its newest in the list, as
   [remove] uncovers the one before it, across the table's growths. *)
let law ops =
  let check i (t, model) =
    let at = Printf.sprintf "after change %d" i in
    equal ~msg:(at ^ ", the bindings") int (List.length model) (T.length t);
    for k = 0 to keys - 1 do
      equal
        ~msg:(Printf.sprintf "%s, key %d" at k)
        (option int) (List.assoc_opt k model) (T.find_opt t k)
    done
  in
  ignore
    (List.fold_left
       (fun (i, state) op ->
         let state = apply state op in
         check i state;
         (i + 1, state))
       (0, (T.create 1, []))
       ops)

let () =
  exit
  @@ run "rig_pci.tables"
    [
      group "tables"
        [
          prop "a table binds each key as a list of its bindings does"
            (Gen.list ~size:(Gen.int_range 0 changes) change)
            law;
        ];
    ]
