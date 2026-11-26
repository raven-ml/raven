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
              (Gen.pair (Gen.of_list [ Opt.Global; Local; Upcast ]) Gen.bool));
         Gen.map
           (fun (axis, amount) -> Opt.Padto { axis; amount })
           (Gen.pair gen_axis gen_small);
         Gen.map
           (fun (axis, with_axis) -> Opt.Swap { axis; with_axis })
           (Gen.pair gen_axis gen_axis);
       ])

(* [o] built anew: equal to it, and another value. *)
let rebuilt : Opt.t -> Opt.t = function
  | Tc t -> Tc { t with axis = t.axis }
  | Split s -> Split { s with axis = s.axis }
  | Padto p -> Padto { p with axis = p.axis }
  | Swap s -> Swap { s with axis = s.axis }

(* Pairs of optimisations, a third of them an optimisation and it rebuilt. *)
let gen_pair =
  Gen.with_pp
    (fun ppf (a, b) -> Format.fprintf ppf "(%a, %a)" Opt.pp a Opt.pp b)
    (Gen.frequency
       [
         (1, Gen.map (fun o -> (o, rebuilt o)) gen_opt);
         (2, Gen.pair gen_opt gen_opt);
       ])

(* Optimisations compared by Opt.equal and ordered by Opt.compare. *)
let by_equal =
  Testable.with_compare Opt.compare (Testable.make ~pp:Opt.pp ~equal:Opt.equal)

let order_of_cell = function
  | "<" -> -1
  | "=" -> 0
  | ">" -> 1
  | o -> failf "no order %s" o

let split target = Opt.Split { axis = 0; amount = 4; target; top = false }

let axes =
  group "axis is the axis an optimisation acts on"
    [
      Golden.cases "reprs.golden" (fun cell ->
          equal int
            (int_of_string (cell "axis"))
            (Opt.axis (Kernel_opts.opt_of_cell (cell "opt"))));
    ]

(* Opt.compare is 0 exactly on structurally equal optimisations, and exactly
   where Opt.equal holds: so Opt.equal is structural equality. *)
let equality =
  prop "equal holds exactly of the same optimisations" gen_pair (fun (a, b) ->
      cover "equal" (a = b);
      Law.order Kernel_opts.opt (a, b, a);
      Law.order by_equal (a, b, b))

let order =
  group "compare"
    [
      group "orders optimisations as comparisons.golden states"
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
      prop "is a total order"
        (Gen.triple gen_opt gen_opt gen_opt)
        (Law.order Kernel_opts.opt);
      cases "orders splits otherwise equal by target: global, local, upcast"
        ~id:(fun (a, b) -> show (split a) ^ " < " ^ show (split b))
        Opt.[ (Global, Local); (Local, Upcast); (Global, Upcast) ]
        (fun (a, b) -> less int ~than:0 (Opt.compare (split a) (split b)));
    ]

let printing =
  group "pp"
    [
      group "formats an optimisation as reprs.golden states"
        [
          Golden.cases "reprs.golden" (fun cell ->
              equal string (cell "opt")
                (show (Kernel_opts.opt_of_cell (cell "opt"))));
        ];
      (* reprs.golden holds no split to a global axis. *)
      cases "formats a split's target as opt.mli states, AxisType and its name"
        ~id:fst
        [
          ("Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.GLOBAL))", Opt.Global);
          ("Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL))", Local);
          ("Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))", Upcast);
        ]
        (fun (expected, target) -> equal string expected (show (split target)));
    ]

let () = exit (Windtrap.run "Tolk.Opt" [ axes; equality; order; printing ])
