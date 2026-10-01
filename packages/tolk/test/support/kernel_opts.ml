open Tolk

(* Optimisations *)

let opt =
  Windtrap.Testable.make ~pp:Opt.pp ~equal:( = )
  |> Windtrap.Testable.with_compare Opt.compare

let target_of_cell = function
  | "AxisType.UPCAST" -> Opt.Upcast
  | "AxisType.UNROLL" -> Unroll
  | "AxisType.LOCAL" -> Local
  | t -> failwith ("no split target " ^ t)

let bool_of_cell = function
  | "True" -> true
  | "False" -> false
  | b -> failwith ("no boolean " ^ b)

let opt_of_cell s =
  let fail () = failwith ("no optimisation " ^ s) in
  match
    Scanf.sscanf s "Opt(op=OptOps.%[A-Z], axis=%d, arg=%[^\n]"
      (fun op axis arg -> (op, axis, String.sub arg 0 (String.length arg - 1)))
  with
  | exception (Scanf.Scan_failure _ | End_of_file | Invalid_argument _) ->
      fail ()
  | op, axis, arg -> (
      let args =
        if String.starts_with ~prefix:"(" arg then
          String.sub arg 1 (String.length arg - 2)
          |> String.split_on_char ',' |> List.map String.trim
        else [ arg ]
      in
      let int = int_of_string in
      match (op, args) with
      | "TC", [ s; o; u ] ->
          Opt.Tc { axis; tc_select = int s; tc_opt = int o; use_tc = int u }
      | "SPLIT", [ a; t ] ->
          Split { axis; amount = int a; target = target_of_cell t; top = false }
      | "SPLIT", [ a; t; b ] ->
          Split
            {
              axis;
              amount = int a;
              target = target_of_cell t;
              top = bool_of_cell b;
            }
      | "PADTO", [ a ] -> Padto { axis; amount = int a }
      | "SWAP", [ w ] -> Swap { axis; with_axis = int w }
      | _ -> fail ())

(* Each optimisation of a tuple starts with "Opt(" and ends at its matching
   parenthesis. *)
let opts_of_cell s =
  let n = String.length s in
  if n < 2 || s.[0] <> '(' || s.[n - 1] <> ')' then
    failwith ("no tuple of optimisations " ^ s);
  let rec close i depth =
    if i >= n then failwith ("unbalanced optimisations " ^ s)
    else
      match s.[i] with
      | '(' -> close (i + 1) (depth + 1)
      | ')' -> if depth = 1 then i else close (i + 1) (depth - 1)
      | _ -> close (i + 1) depth
  in
  let rec opts i =
    match String.index_from_opt s i 'O' with
    | None -> []
    | Some start ->
        let stop = close (String.index_from s start '(') 0 in
        opt_of_cell (String.sub s start (stop - start + 1)) :: opts (stop + 1)
  in
  opts 1

(* Settings *)

let settings_of_cell s =
  let setting pair =
    match String.split_on_char '=' pair with
    | [ "TC"; v ] -> Helpers.B (Helpers.use_tc, int_of_string v)
    | [ "TC_OPT"; v ] -> B (Helpers.tc_opt, int_of_string v)
    | [ "TC_SELECT"; v ] -> B (Helpers.tc_select, int_of_string v)
    | [ "TC_MIN_GLOBALS"; v ] -> B (Helpers.tc_min_globals, int_of_string v)
    | [ "ALLOW_TF32"; v ] -> B (Helpers.allow_tf32, int_of_string v <> 0)
    | [ "NOOPT"; v ] -> B (Helpers.noopt, int_of_string v <> 0)
    | [ "EMULATED_DTYPES"; v ] ->
        B (Helpers.emulated_dtypes, String.split_on_char ',' v)
    | [ "DISABLE_FAST_IDIV"; v ] ->
        B (Helpers.disable_fast_idiv, int_of_string v <> 0)
    | [ "TRANSCENDENTAL"; v ] -> B (Helpers.transcendental, int_of_string v)
    | _ -> failwith ("no setting " ^ pair)
  in
  String.split_on_char ' ' s |> List.filter (( <> ) "") |> List.map setting

(* Renderers *)

let renderer_of_row cell =
  let device = cell "device" and arch = cell "arch" in
  let tensor_cores =
    match device with
    | "METAL" -> Tc.metal
    | "CUDA" | "NV" -> Tc.cuda arch
    | "AMD" -> Tc.amd arch
    | _ -> []
  in
  if List.length tensor_cores <> int_of_string (cell "tensor_cores") then
    failwith ("the tensor cores of " ^ device ^ " " ^ arch ^ " differ in number");
  Renderer.v
    ~has_local:(bool_of_cell (cell "has_local"))
    ~has_shared:(bool_of_cell (cell "has_shared"))
    ~shared_max:(int_of_string (cell "shared_max"))
    ~tensor_cores
    (Helpers.target ~arch device)

(* Writes *)

let inputs k =
  let contents (p : Ops.param_arg) size =
    Array.init size (fun i ->
        if Dtype.is_bool p.dtype then `Bool (i mod 2 = 0)
        else if Dtype.is_float p.dtype then `Float (Float.of_int (i mod 3))
        else `Int (Bigint.of_int (i mod 3)))
  in
  Ops.toposort k
  |> List.filter_map (fun u ->
      match (Ops.op u, Ops.arg u) with
      | Op.Param, Param ({ slot; size = Some size; _ } as p) when slot >= 0 ->
          Some (slot, contents p size)
      | _ -> None)
  |> List.sort_uniq (fun (s0, _) (s1, _) -> Int.compare s0 s1)

let variables k =
  List.map
    (fun v ->
      match Ops.vmax v with
      | `Int z -> (Ops.expr v, Bigint.to_int z)
      | _ -> invalid_arg ("variable " ^ Ops.expr v ^ " is no integer"))
    (Ops.variables k)

let writes k =
  let vars = List.map (fun v -> (Ops.expr v, Ops.vmax v)) (Ops.variables k) in
  Interpreter.writes ~vars ~buffers:(inputs k) k

let close_values v0 v1 =
  match (v0, v1) with
  | `Float f0, `Float f1 ->
      Float.equal f0 f1
      || Float.abs (f0 -. f1) <= 1e-5 *. Float.max (Float.abs f0) (Float.abs f1)
  | v0, v1 -> Windtrap.Testable.equal Dtypes.value v0 v1

let write =
  let pp ppf (slot, i, v) =
    Format.fprintf ppf "slot %d [%d] = %a" slot i
      (Windtrap.Testable.pp Dtypes.value)
      v
  in
  Windtrap.Testable.make ~pp ~equal:(fun (s0, i0, v0) (s1, i1, v1) ->
      s0 = s1 && i0 = i1 && close_values v0 v1)
