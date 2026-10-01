open Windtrap
open Tolk

let show o = Format.asprintf "%a" Opt.pp o
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
              equal string (cell "opt")
                (show (Kernel_opts.opt_of_cell (cell "opt"))));
        ];
      group "axis is the axis tinygrad prints"
        [
          Golden.cases "reprs.golden" (fun cell ->
              equal int
                (int_of_string (cell "axis"))
                (Opt.axis (Kernel_opts.opt_of_cell (cell "opt"))));
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
                      (Kernel_opts.opt_of_cell (cell "a"))
                      (Kernel_opts.opt_of_cell (cell "b")))
                   0));
        ];
      prop "compare is a total order"
        (Gen.triple gen_opt gen_opt gen_opt)
        (Law.order Kernel_opts.opt);
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

let () = exit (Windtrap.run "Tolk.Opt" [ printing; order ])
