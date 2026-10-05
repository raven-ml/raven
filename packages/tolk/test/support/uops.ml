open Tolk

let pp_uop ppf u =
  let lines = String.split_on_char '\n' (String.trim (Graph.to_string u)) in
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list Format.pp_print_string)
    lines

let uop =
  Windtrap.Testable.with_compare Ops.compare
    (Windtrap.Testable.make ~pp:pp_uop ~equal:Ops.equal)

(* [substituted u subs] is [u] with [subs] applied, the storage that the queue
   data of its calls names included. *)
let substituted u subs =
  let u = Ops.substitute ~calls:Enter u subs in
  let mapped n = Option.value (List.assq_opt n subs) ~default:n in
  let requeued c =
    match Ops.arg c with
    | Call ({ aux = Some info; _ } as ci) ->
        let inputs = List.map (fun (b, o, d) -> (mapped b, o, d)) info.inputs in
        let written_bufs = List.map mapped info.written_bufs in
        let writes = List.map mapped info.writes in
        let aux = Some { info with inputs; written_bufs; writes } in
        Some (c, Ops.replace ~arg:(Call { ci with aux }) c)
    | _ -> None
  in
  Ops.substitute ~calls:Enter u
    (List.filter_map requeued (Ops.toposort ~calls:Enter u))

let numbered_like like u =
  let made op g =
    List.filter (fun n -> Ops.op n = op) (Ops.toposort ~calls:Enter g)
  in
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
  | subs -> substituted u subs
  | exception Invalid_argument _ -> u

let binaries_as_sources u =
  let recorded prg =
    match List.rev (Ops.src prg) with
    | binary :: source :: _ when Ops.op binary = Binary -> (
        match Ops.arg source with
        | String text -> Some (binary, Ops.v Binary ~arg:(Bytes text))
        | _ -> None)
    | _ -> None
  in
  let programs =
    List.filter (fun n -> Ops.op n = Program) (Ops.toposort ~calls:Enter u)
  in
  Ops.substitute ~calls:Enter u (List.filter_map recorded programs)

let placeholders_like like u =
  let placeholders g =
    List.filter
      (fun n -> Ops.op n = Param && Option.is_some (Ops.tag n))
      (Ops.toposort ~calls:Enter g)
  in
  let renumbered mine theirs =
    match (Ops.arg mine, Ops.arg theirs) with
    | Param p, Param q ->
        (mine, Ops.replace ~arg:(Param { p with slot = q.slot }) mine)
    | _ -> (mine, mine)
  in
  match List.map2 renumbered (placeholders u) (placeholders like) with
  | subs -> substituted u subs
  | exception Invalid_argument _ -> u

let without_profile_keys u =
  let unkeyed c =
    match Ops.arg c with
    | Call ({ aux = Some info; _ } as ci) ->
        let kernels =
          List.map
            (fun (k : Ops.hcq_kernel) -> { k with profile_key = None })
            info.kernels
        in
        Some
          ( c,
            Ops.replace
              ~arg:(Call { ci with aux = Some { info with kernels } })
              c )
    | _ -> None
  in
  Ops.substitute ~calls:Skip u
    (List.filter_map unkeyed (Ops.toposort ~calls:Enter u))
