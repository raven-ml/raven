open Windtrap
open Tolk

let rejects f = raises_match (Exn.invalid_arg ?sub:None) f
let uop = Uops.uop
let uops = list Uops.uop

let estimates =
  let equal (e0 : Ops.estimates) (e1 : Ops.estimates) =
    Shape.Sint.equal e0.ops e1.ops
    && Shape.Sint.equal e0.lds e1.lds
    && Shape.Sint.equal e0.mem e1.mem
  in
  Testable.make ~pp:Ops.pp_estimates ~equal

let counts ops lds mem : Ops.estimates =
  { ops = Int ops; lds = Int lds; mem = Int mem }

(* Targets *)

let target ?(arch = "x86_64,x86-64") device renderer =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

let clang = target "CPU" ""
let targets _ = clang

(* A renderer whose binary is its source's bytes, for the calls built here. *)
let uncompiled =
  Renderer.with_compiler (Renderer.Compiler.v Fun.id) (Cstyle.clang clang)

(* Kernels and calls *)

let buffer ?(device = "CPU") ?(dtype = Dtype.Float32) slot n =
  Shape.param ~shape:[ Int n ] ~device slot dtype

(* The kernel that stores [f] of each element of slot [src] (default [1]) into
   slot 0. *)
let map_kernel ?(name = "map") ?(beam = 0) ?(n = 4) ?device ?(src = 1) f =
  let out = buffer ?device 0 n and inp = buffer ?device src n in
  let i = Ops.range (Int n) [ 0 ] in
  let store = Ops.store (Ops.index out [ i ]) (f (Ops.index inp [ i ])) in
  Ops.sink ~kernel:(Ops.kernel_info ~name ~beam ()) [ Ops.end_ store [ i ] ]

let plus_one = map_kernel (fun x -> Ops.add x (Ops.float ~dtype:Float32 1.))

(* The kernel that stores [x * n] into slot 0, [n] a variable read by name. *)
let scaled_kernel variables =
  let out = buffer 0 1 in
  let product =
    List.fold_left
      (fun acc v -> Ops.mul acc (Ops.cast v Dtype.Float32))
      (Ops.float ~dtype:Float32 2.)
      variables
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"scaled" ())
    [ Ops.store (Ops.index out [ Ops.int 0 ]) product ]

let program kernel = Codegen.to_program kernel uncompiled
let n = Ops.variable ~dtype:Int32 "n" (`Int Bigint.one) (`Int (Bigint.of_int 8))
let m = Ops.variable ~dtype:Int32 "m" (`Int Bigint.one) (`Int (Bigint.of_int 8))
let bound v x = Shape.bind v (`Int (Bigint.of_int x))
let loop = Ops.range ~axis_type:Loop (Int 4) [ 100 ]

let storage ?(device = "CPU") ?(dtype = Dtype.Float32) n =
  Ops.new_buffer device n dtype

let linear calls = Ops.v Linear ~src:calls
let custom_call args = Ops.call (Ops.custom_function "f" []) args

let program_info prg =
  match Ops.arg prg with
  | Program p -> p
  | _ -> invalid_arg "a program holds its program information"

let calls_of l = List.map Ops.without_after (Ops.src l)
let body_of call = Ops.nth call 0

(* The graph a compiled schedule is compared with: each program's binary its
   source's bytes, as the goldens record them. *)
let as_recorded linear = Uops.binaries_as_sources linear

(* Recorded schedules *)

let programs =
  [
    "add";
    "sum";
    "two_outputs";
    "assign";
    "custom_kernel";
    "custom_kernel_scalar";
    "precompiled_scalar";
    "copy";
    "copy_view";
    "copy_big";
    "copy_variable";
    "variable_shrink";
    "variable_two";
  ]

let schedule name = Golden.sink (name ^ ".golden")
let compiled name = Golden.sink (name ^ "_compiled.golden")

(* A cell's values separated by spaces, [-] for none. *)
let words cell = if cell = "-" then [] else String.split_on_char ' ' cell

let var_vals cell =
  List.map
    (fun w ->
      match String.split_on_char '=' w with
      | [ k; v ] -> (k, int_of_string v)
      | _ -> invalid_arg ("no binding " ^ w))
    (words cell)

let position call u =
  let rec find i = function
    | s :: _ when s == u -> Printf.sprintf "%%%d" i
    | _ :: rest -> find (i + 1) rest
    | [] -> Format.asprintf "%a" Ops.pp_arg (Ops.arg u)
  in
  find 0 (Ops.src call)

(* A variable by its name, a bound value by its value. *)
let variable u =
  match Ops.arg u with
  | Param { name = Some name; _ } -> name
  | Const c -> Format.asprintf "%a" Dtype.pp_const c
  | _ -> Format.asprintf "%a" Ops.pp_arg (Ops.arg u)

(* A kernel argument: a buffer as its position in [call], a scalar as
   {!variable} says. *)
let kernel_arg call u =
  if Ops.addrspace u = Some Alu || Ops.op u = Const then variable u
  else position call u

let recorded_call cell =
  List.nth (Ops.src (compiled (cell "program"))) (int_of_string (cell "call"))

let in_colour f = Setting.context [ B (Setting.no_color, false) ] f

(* The goldens were recorded without colour, which kernel names carry. *)
let plain f = Setting.context [ B (Setting.no_color, true) ] f

let reads_as_recorded cell =
  let call = recorded_call cell and var_vals = var_vals (cell "var_vals") in
  let bufs = Realize.get_call_arg_uops call in
  let body = body_of call in
  let outs, ins = Realize.get_call_outs_ins call in
  let e = Realize.estimate_uop call in
  let ints = List.map string_of_int in
  equal (list string) ~msg:"bufs"
    (words (cell "bufs"))
    (List.map (position call) bufs);
  if Op.equal (Ops.op body) Program then
    equal (list string) ~msg:"args"
      (words (cell "args"))
      (List.map (kernel_arg call) (Realize.get_call_kernel_args call body));
  equal (list string) ~msg:"outs" (words (cell "outs")) (ints outs);
  equal (list string) ~msg:"ins" (words (cell "ins")) (ints ins);
  equal string ~msg:"name" (cell "name")
    (in_colour (fun () -> Realize.get_call_name ~var_vals call bufs));
  equal (list string) ~msg:"estimates"
    [ cell "ops"; cell "lds"; cell "mem" ]
    (ints
       (List.map (fun s -> Shape.sym_infer s var_vals) [ e.ops; e.lds; e.mem ]))

let recorded_calls =
  group "calls of recorded schedules"
    [ Golden.cases "calls.golden" ~key:[ "program"; "call" ] reads_as_recorded ]

(* Calls *)

let arguments =
  group "get_call_arg_uops"
    [
      test "is a call's storage arguments in order, without its scalar ones"
        (fun () ->
          let a = storage 4 and b = storage 4 in
          let trip = Ops.O.((loop * int 2) + int 1) in
          let call =
            Ops.call
              (program (scaled_kernel [ n ]))
              [ a; bound m 3; b; n; trip ]
          in
          equal uops [ a; b ] (Realize.get_call_arg_uops call));
      test "is empty for a call of no argument" (fun () ->
          equal uops []
            (Realize.get_call_arg_uops (Ops.call (Ops.sink []) [])));
    ]

let kernel_args =
  group "get_call_kernel_args"
    [
      test
        "is the constant a call binds each variable to, in the program's order"
        (fun () ->
          let prg = program (scaled_kernel [ n; m ]) in
          let a = storage 1 in
          let call = Ops.call prg [ a; bound m 5; bound n 2 ] in
          equal (list string) ~msg:"the program's parameters"
            [ "b0"; "n"; "m" ]
            (List.map
               (fun p ->
                 if Ops.addrspace p = Some Alu then variable p else "b0")
               (Shape.kernel_params prg));
          equal uops
            [ a; Ops.int 2; Ops.int 5 ]
            (Realize.get_call_kernel_args call prg));
      test "is the variable itself, at no slot, where the call leaves it free"
        (fun () ->
          let prg = program (scaled_kernel [ n; m ]) in
          let call = Ops.call prg [ storage 1; bound m 5 ] in
          match Realize.get_call_kernel_args call prg with
          | [ _; free; five ] ->
              equal string "n" (variable free);
              equal int (-1)
                (match Ops.arg free with Param p -> p.slot | _ -> -2);
              equal uop (Ops.int 5) five
          | us -> failf "three values, not %d" (List.length us));
      test "is the call's argument in the slot of a scalar parameter" (fun () ->
          let index = Shape.param ~addrspace:Alu 1 Int32 in
          let prg = program (scaled_kernel [ n; index ]) in
          let trip = Ops.O.((loop * int 2) + int 1) and a = storage 1 in
          let call = Ops.call prg [ a; trip; bound n 4 ] in
          equal uops [ a; trip; Ops.int 4 ]
            (Realize.get_call_kernel_args call prg));
      test "is each buffer's argument for a program of no variable" (fun () ->
          let prg = program (scaled_kernel []) and a = storage 1 in
          equal uops [ a ]
            (Realize.get_call_kernel_args (Ops.call prg [ a ]) prg));
    ]

let outs_ins =
  group "get_call_outs_ins"
    [
      test "is a program's outputs and inputs" (fun () ->
          let prg = program plus_one in
          let info = program_info prg in
          equal
            (pair (list int) (list int))
            (info.outs, info.ins)
            (Realize.get_call_outs_ins (Ops.call prg [ storage 4; storage 4 ])));
      test "is the destination written and the source read for a copy"
        (fun () ->
          equal
            (pair (list int) (list int))
            ([ 0 ], [ 1 ])
            (Realize.get_call_outs_ins (Shape.store_call (storage 4) (storage 4))));
      test "is nothing for any other call" (fun () ->
          equal
            (pair (list int) (list int))
            ([], [])
            (Realize.get_call_outs_ins (custom_call [ storage 4 ])));
    ]

let names =
  group "get_call_name"
    [
      test "is a program's kernel name, as the kernel gives it" (fun () ->
          let call =
            Ops.call
              (program (map_kernel ~name:"plus one" Fun.id))
              [ storage 4; storage 4 ]
          in
          equal string "plus one"
            (Realize.get_call_name call (Realize.get_call_arg_uops call)));
      test "is a copy's size and devices, each cut to seven characters"
        (fun () ->
          let dst = storage ~device:"METAL:12345" 4
          and src = storage ~device:"CPU" 4 in
          let call = Shape.store_call dst src in
          equal string "copy       16 B, METAL:1 <- CPU    "
            (plain (fun () ->
                 Realize.get_call_name call (Realize.get_call_arg_uops call))));
      test "is in yellow" (fun () ->
          let call = Shape.store_call (storage 4) (storage 4) in
          equal string
            (Helpers.colored Yellow "copy       16 B,     CPU <- CPU    ")
            (in_colour (fun () ->
                 Realize.get_call_name call (Realize.get_call_arg_uops call))));
      test "sizes a copy with the variables' values" (fun () ->
          let v =
            Ops.variable "v" (`Int Bigint.one) (`Int (Bigint.of_int 10))
          in
          let src = Shape.shrink (storage 10) [ Some (Int 0, Sym v) ] in
          let call =
            Shape.store_call (Shape.shrink (storage 10) [ Some (Int 0, Sym v) ]) src
          in
          equal string "copy       12 B,     CPU <- CPU    "
            (plain (fun () ->
                 Realize.get_call_name
                   ~var_vals:[ ("v", 3) ]
                   call
                   (Realize.get_call_arg_uops call))));
      test "raises Invalid_argument for any other call" (fun () ->
          let call = custom_call [ storage 4 ] in
          rejects (fun () ->
              Realize.get_call_name call (Realize.get_call_arg_uops call)));
    ]

let costs =
  group "estimate_uop"
    [
      test "is a program's kernel estimates" (fun () ->
          let prg = program plus_one in
          let expected =
            match Ops.arg (Ops.nth prg 0) with
            | Kernel { estimates = Some e; _ } -> e
            | _ -> fail "a compiled kernel holds its estimates"
          in
          equal estimates expected
            (Realize.estimate_uop (Ops.call prg [ storage 4; storage 4 ])));
      test
        "is nothing for a program whose kernel holds no estimates, as \
         tinygrad's" (fun () ->
          let prg = program plus_one in
          let bare = Ops.replace ~src:(plus_one :: List.tl (Ops.src prg)) prg in
          equal estimates (counts 0 0 0)
            (Realize.estimate_uop (Ops.call bare [ storage 4; storage 4 ])));
      test "sees a call through its afters" (fun () ->
          let call = Shape.store_call (storage 4) (storage 4) in
          equal estimates
            (Realize.estimate_uop call)
            (Realize.estimate_uop (Ops.after call [ storage 1 ])));
      test "is a copy's bytes, loaded and stored and touched" (fun () ->
          equal estimates (counts 0 24 24)
            (Realize.estimate_uop
               (Shape.store_call (storage ~dtype:Int16 12)
                  (storage ~dtype:Int16 12))));
      test "is nothing for any other call" (fun () ->
          equal estimates (counts 0 0 0)
            (Realize.estimate_uop (custom_call [ storage 4 ])));
    ]

(* Compiling *)

let recorded_compilation =
  group "lower_and_compile of recorded schedules"
    (List.map
       (fun name ->
         Golden.graph (name ^ "_compiled.golden") (fun () ->
             plain (fun () ->
                 as_recorded
                   (Realize.lower_and_compile ~targets (schedule name)))))
       programs)

let compiled_bodies l = List.map (fun c -> Ops.op (body_of c)) (calls_of l)

let compiling =
  group "lower_and_compile"
    [
      test "leaves a linear of no kernel as it is" (fun () ->
          let l = linear [ Shape.store_call (storage 4) (storage 4) ] in
          is_true (Realize.lower_and_compile ~targets l == l));
      test "makes each call of a kernel a call of its program" (fun () ->
          let l = linear [ Ops.call plus_one [ storage 4; storage 4 ] ] in
          equal
            (list (Testable.make ~pp:Op.pp ~equal:Op.equal))
            [ Program ]
            (compiled_bodies (Realize.lower_and_compile ~targets l)));
      test "compiles each kernel once" (fun () ->
          let l =
            linear
              [
                Ops.call plus_one [ storage 4; storage 4 ];
                Ops.call plus_one [ storage 4; storage 4 ];
              ]
          in
          match calls_of (Realize.lower_and_compile ~targets l) with
          | [ c0; c1 ] -> equal uop (body_of c0) (body_of c1)
          | cs -> failf "two calls, not %d" (List.length cs));
      test "compiles the kernels of one key once, each on its call's device"
        (fun () ->
          let fresh = float_of_int (Ops.unique_num ()) in
          let call d =
            let scale x = Ops.mul x (Ops.float ~dtype:Float32 fresh) in
            let kernel = map_kernel ~device:d scale in
            Ops.call kernel [ storage ~device:d 4; storage ~device:d 4 ]
          in
          let devices c =
            List.sort_uniq String.compare
              (List.filter_map
                 (fun u ->
                   match Ops.arg u with
                   | Ops.Param { device = Some d; _ } -> Some d
                   | _ -> None)
                 (Ops.toposort ~calls:Enter (body_of c)))
          in
          let before = Codegen.programs () in
          let l =
            Realize.lower_and_compile ~targets
              (linear [ call "CPU:1"; call "CPU:2" ])
          in
          equal int ~msg:"programs made" 1 (Codegen.programs () - before);
          equal (list (list string)) [ [ "CPU:1" ]; [ "CPU:2" ] ]
            (List.map devices (calls_of l)));
      test "collects each kernel's name and key, made or kept, on its domain"
        (fun () ->
          let fresh = float_of_int (Ops.unique_num ()) in
          let kernel d =
            let scale x = Ops.mul x (Ops.float ~dtype:Float32 fresh) in
            map_kernel ~device:d scale
          in
          let sinks = [ kernel "CPU:1"; kernel "CPU:2" ] in
          let call k d =
            Ops.call k [ storage ~device:d 4; storage ~device:d 4 ]
          in
          let compile () =
            ignore
              (Realize.lower_and_compile ~targets
                 (linear (List.map2 call sinks [ "CPU:1"; "CPU:2" ])))
          in
          let kernels = list (pair string string) in
          let found f =
            let v, ks = Realize.kernels f in
            (v, List.map (fun (k : Realize.kernel) -> (k.name, k.key)) ks)
          in
          let (), made = found compile in
          let ((), kept), outer = found (fun () -> found compile) in
          let (), elsewhere =
            found (fun () -> Domain.join (Domain.spawn compile))
          in
          let (), all = Realize.kernels compile in
          equal (list uop) ~msg:"each kernel's sink" sinks
            (List.map (fun (k : Realize.kernel) -> k.ast) all);
          (* Each kernel's optimisations are those its kept program applied. *)
          List.iter
            (fun (k : Realize.kernel) ->
              match Codegen.made k.ast k.renderer with
              | None -> failf "%s has no program" k.name
              | Some prg -> (
                  match Ops.arg (Ops.nth prg 0) with
                  | Kernel i ->
                      equal (list Kernel_opts.opt) ~msg:k.name i.applied_opts
                        k.opts
                  | _ -> failf "%s's program names no kernel" k.name))
            all;
          equal int ~msg:"kernels" 2 (List.length made);
          equal int ~msg:"keys" 1
            (List.length (List.sort_uniq String.compare (List.map snd made)));
          equal kernels ~msg:"kept" made kept;
          equal kernels ~msg:"around an inner collector" [] outer;
          equal kernels ~msg:"on another domain" [] elsewhere);
      test "keeps each call's arguments" (fun () ->
          let args = [ storage 4; storage 4 ] in
          match
            calls_of
              (Realize.lower_and_compile ~targets
                 (linear [ Ops.call plus_one args ]))
          with
          | [ c ] -> equal uops args (List.tl (Ops.src c))
          | cs -> failf "one call, not %d" (List.length cs));
      test "leaves a compiled program alone" (fun () ->
          let l =
            linear [ Ops.call (program plus_one) [ storage 4; storage 4 ] ]
          in
          equal uop l (Realize.lower_and_compile ~targets l));
      test "raises Invalid_argument when no renderer serves a call's target"
        (fun () ->
          rejects (fun () ->
              Realize.lower_and_compile
                ~targets:(fun _ -> target "NULL" "")
                (linear [ Ops.call plus_one [ storage 4; storage 4 ] ])));
    ]

(* Beam search *)

(* A search that records the width each kernel asks for and leaves the kernel as
   it is, with the most searches it saw run at once. *)
let recording_search () =
  let widths = ref [] and live = Atomic.make 0 and peak = Atomic.make 0 in
  let lock = Mutex.create () in
  let search width s =
    let now = 1 + Atomic.fetch_and_add live 1 in
    if now > Atomic.get peak then Atomic.set peak now;
    Mutex.protect lock (fun () -> widths := width :: !widths);
    Unix.sleepf 0.01;
    Atomic.decr live;
    s
  in
  (search, (fun () -> List.rev !widths), fun () -> Atomic.get peak)

(* Kernels no other call compiles, even in a rerun of the suite, since compiled
   programs are kept. *)
let beam_kernels () =
  let fresh = float_of_int (Ops.unique_num ()) in
  List.init 4 (fun i ->
      map_kernel ~name:(Printf.sprintf "k%d" i) ~beam:(i + 2) (fun x ->
          Ops.mul x (Ops.float ~dtype:Float32 (fresh +. float_of_int i))))

let beam =
  group "lower_and_compile with a beam search"
    [
      test
        "searches each kernel with the width it asks for, in order, one at a \
         time" (fun () ->
          let search, widths, peak = recording_search () in
          let calls =
            List.map
              (fun k -> Ops.call k [ storage 4; storage 4 ])
              (beam_kernels ())
          in
          ignore (Realize.lower_and_compile ~search ~targets (linear calls));
          equal (list int) [ 2; 3; 4; 5 ] (widths ());
          equal int ~msg:"searches at once" 1 (peak ()));
      test
        "raises Invalid_argument for a kernel that asks for one when none is \
         given" (fun () ->
          rejects (fun () ->
              Realize.lower_and_compile ~targets
                (linear
                   [
                     Ops.call
                       (map_kernel ~name:"asks" ~beam:2 Fun.id)
                       [ storage 4; storage 4 ];
                   ])));
    ]

(* Parallel compilation *)

let distinct_kernels =
  List.init 6 (fun i ->
      map_kernel ~name:(Printf.sprintf "p%d" i) (fun x ->
          Ops.add x (Ops.float ~dtype:Float32 (float_of_int (100 + i)))))

let parallel =
  group "lower_and_compile in parallel"
    [
      test "compiles each kernel into the program it compiles into alone"
        (fun () ->
          let calls =
            List.map
              (fun k -> Ops.call k [ storage 4; storage 4 ])
              distinct_kernels
          in
          let alone =
            List.map
              (fun k ->
                Codegen.to_program k (Device.renderer clang |> Result.get_ok))
              distinct_kernels
          in
          equal uops alone
            (List.map body_of
               (calls_of (Realize.lower_and_compile ~targets (linear calls)))));
    ]

let () =
  exit
    (run "Realize"
       [
         recorded_calls;
         arguments;
         kernel_args;
         outs_ins;
         names;
         costs;
         recorded_compilation;
         compiling;
         beam;
         parallel;
       ])
