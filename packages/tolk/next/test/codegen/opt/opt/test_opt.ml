open Windtrap
open Tolk_next

let show o = Format.asprintf "%a" Opt.pp o

let opt =
  Testable.make ~pp:Opt.pp ~equal:( = ) |> Testable.with_compare Opt.compare

(* Reads an Opt as tinygrad prints it: [Opt(op=OptOps.SPLIT, axis=0, arg=(4,
   AxisType.UPCAST))]. *)
let opt_of_repr s =
  let target = function
    | "AxisType.UPCAST" -> Opt.Upcast
    | "AxisType.UNROLL" -> Unroll
    | "AxisType.LOCAL" -> Local
    | t -> failf "no split target %s" t
  in
  let top = function
    | "True" -> true
    | "False" -> false
    | b -> failf "no boolean %s" b
  in
  Scanf.sscanf s "Opt(op=OptOps.%[A-Z], axis=%d, arg=%s@\n" (fun op axis arg ->
      let arg = String.sub arg 0 (String.length arg - 1) in
      let parts =
        if String.starts_with ~prefix:"(" arg then
          String.split_on_char ',' (String.sub arg 1 (String.length arg - 2))
          |> List.map String.trim
        else [ arg ]
      in
      match (op, parts) with
      | "TC", [ s; o; u ] ->
          Opt.Tc
            {
              axis;
              tc_select = int_of_string s;
              tc_opt = int_of_string o;
              use_tc = int_of_string u;
            }
      | "SPLIT", [ a; t ] ->
          Split
            { axis; amount = int_of_string a; target = target t; top = false }
      | "SPLIT", [ a; t; b ] ->
          Split
            { axis; amount = int_of_string a; target = target t; top = top b }
      | "PADTO", [ a ] -> Padto { axis; amount = int_of_string a }
      | "SWAP", [ w ] -> Swap { axis; with_axis = int_of_string w }
      | _ -> failf "no optimisation %s" s)

let gen_axis = Gen.int_range (-1) 3
let gen_small = Gen.int_range (-1) 4

let gen_opt =
  Gen.with_pp Opt.pp
    (Gen.one_of
       [
         Gen.map
           (fun (axis, (tc_select, tc_opt, use_tc)) ->
             Opt.Tc { axis; tc_select; tc_opt; use_tc })
           (Gen.pair gen_axis (Gen.triple gen_small gen_small gen_small));
         Gen.map
           (fun ((axis, amount), (target, top)) ->
             Opt.Split { axis; amount; target; top })
           (Gen.pair
              (Gen.pair gen_axis gen_small)
              (Gen.pair (Gen.of_list [ Opt.Upcast; Unroll; Local ]) Gen.bool));
         Gen.map
           (fun (axis, amount) -> Opt.Padto { axis; amount })
           (Gen.pair gen_axis gen_small);
         Gen.map
           (fun (axis, with_axis) -> Opt.Swap { axis; with_axis })
           (Gen.pair gen_axis gen_axis);
       ])

let order_of_cell = function
  | "<" -> -1
  | "=" -> 0
  | ">" -> 1
  | o -> failf "no order %s" o

(* Printing *)

let printing =
  group "printing"
    [
      group "pp is tinygrad's repr"
        [
          Golden.cases "reprs.golden" (fun cell ->
              equal string (cell "opt") (show (opt_of_repr (cell "opt"))));
        ];
      group "axis is the axis tinygrad prints"
        [
          Golden.cases "reprs.golden" (fun cell ->
              equal int
                (int_of_string (cell "axis"))
                (Opt.axis (opt_of_repr (cell "opt"))));
        ];
      test "a split that is not from the top prints without its flag" (fun () ->
          equal string "Opt(op=OptOps.SPLIT, axis=2, arg=(0, AxisType.UNROLL))"
            (show
               (Split { axis = 2; amount = 0; target = Unroll; top = false })));
    ]

(* Order *)

let order =
  group "order"
    [
      group "compare agrees with tinygrad where tinygrad orders"
        [
          Golden.cases "comparisons.golden" ~key:[ "a"; "b" ] (fun cell ->
              equal int
                (order_of_cell (cell "order"))
                (Int.compare
                   (Opt.compare
                      (opt_of_repr (cell "a"))
                      (opt_of_repr (cell "b")))
                   0));
        ];
      prop "compare is a total order"
        (Gen.triple gen_opt gen_opt gen_opt)
        (Law.order opt);
      prop "compare is 0 exactly on equal optimisations"
        (Gen.pair gen_opt gen_opt) (fun (o0, o1) ->
          equal bool (o0 = o1) (Opt.compare o0 o1 = 0));
      test "splits into different targets are different" (fun () ->
          let split target =
            Opt.Split { axis = 0; amount = 4; target; top = false }
          in
          not_equal int 0 (Opt.compare (split Upcast) (split Unroll));
          not_equal int 0 (Opt.compare (split Unroll) (split Local)));
    ]

(* Errors *)

let errors =
  group "check"
    [
      test "is unit when its condition holds" (fun () ->
          Opt.check true "unused");
      test "raises its message when its condition fails" (fun () ->
          raises (Opt.Kernel_opt_error "padto arg is a multiple > 1, not 1")
            (fun () -> Opt.check false "padto arg is a multiple > 1, not 1"));
    ]

let () = exit (Windtrap.run "Tolk_next.Opt" [ printing; order; errors ])
