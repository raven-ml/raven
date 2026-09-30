open Tolk_next

let pp_uop ppf u =
  let lines = String.split_on_char '\n' (String.trim (Graph.to_string u)) in
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list Format.pp_print_string)
    lines

let uop =
  Windtrap.Testable.with_compare Ops.compare
    (Windtrap.Testable.make ~pp:pp_uop ~equal:Ops.equal)

let numbered_like like u =
  let made op g = List.filter (fun n -> Ops.op n = op) (Ops.toposort g) in
  let renumbered mine theirs =
    match (Ops.arg mine, Ops.arg theirs) with
    | Param p, Param q ->
        (mine, Ops.replace ~arg:(Param { p with slot = q.slot }) mine)
    | _ -> (mine, mine)
  in
  match
    List.concat_map
      (fun op -> List.map2 renumbered (made op u) (made op like))
      Op.[ Alloc; Buffer ]
  with
  | subs -> Ops.substitute ~enter_calls:true u subs
  | exception Invalid_argument _ -> u
