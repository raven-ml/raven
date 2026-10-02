(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Transformations of compiled functions. A transformation of a compiled
   function agrees with the transformation of the function: grad, vjp, jvp,
   vmap, a total collected around them and their compositions, over shapes with
   no element and two dtypes, for functions whose backward pass reads computed
   values, arguments, movements of arguments and captures, constants, a remat,
   a scan and a map's collectives. Reverse mode through a compiled call fixes
   its residuals, of which no capture or movement of an argument is one, and
   traces its forward pass once per key and set of tracked arguments, traces its
   backward pass once per layout of the cotangents, and never runs the function
   there. A compiled function refuses, under every transformation, a value the
   transformation tracks that it reads through its closure, and refuses a traced
   value that escaped the trace that made it. *)

open Windtrap

(* Inputs *)

type input = Input : (float, 'b) Nx.dtype * (float, 'b) Nx.t -> input

let pp_input ppf (Input (dt, x)) =
  Format.fprintf ppf "%a %a" Nx.pp_dtype dt Nx.pp x

let values dt shape seed =
  let st = Random.State.make [| seed |] in
  Nx.init dt shape (fun _ -> Random.State.float st 3. -. 1.5)

let shapes =
  [ [| 3 |]; [| 0 |]; [| 1 |]; [| 2; 3 |]; [| 4; 0 |]; [| 0; 2 |]; [| 1; 5 |] ]

(* Shapes whose rows a map takes, and those a scan takes: at least one row. A
   map of a scan takes lanes of those, and over no lane its carry has no element
   while its step reads rows that have some. *)
let matrices = List.filter (fun s -> Array.length s = 2) shapes
let rows = List.filter (fun s -> s.(0) > 0) matrices

let lanes_of_rows =
  [ [| 2; 3; 2 |]; [| 3; 1; 4 |]; [| 2; 2; 0 |]; [| 0; 2; 3 |] ]

(* Each shape with no element, at each dtype: every law runs them. *)
let empty shapes =
  List.concat_map
    (fun shape ->
      if Array.fold_left ( * ) 1 shape > 0 then []
      else
        [
          Input (Nx.float32, values Nx.float32 shape 0);
          Input (Nx.float64, values Nx.float64 shape 0);
        ])
    shapes

let input shapes =
  let open Gen in
  with_pp pp_input
    ( bind (of_list [ `Float32; `Float64 ]) @@ fun dt ->
      bind (of_list ~pp:Nx.pp_shape shapes) @@ fun shape ->
      map
        (fun seed ->
          match dt with
          | `Float32 -> Input (Nx.float32, values Nx.float32 shape seed)
          | `Float64 -> Input (Nx.float64, values Nx.float64 shape seed))
        (int_range 0 1_000_000) )

(* Compiled float results differ from eager's in rounding, and float32
   transcendentals within a few units in the last place: a second derivative
   that cancels terms of magnitude near 10 keeps an absolute error near 1e-5. *)
let close (type b) (dt : (float, b) Nx.dtype) : (float, b) Nx.t testable =
  match dt with
  | Float32 -> Oracle.tensor ~rel:1e-4 ~abs:1e-4 ()
  | _ -> Oracle.tensor ~rel:1e-9 ~abs:1e-12 ()

(* Functions *)

type fn = { name : string; f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

let w64 = values Nx.float64 [| 64 |] 7
let w32 = Nx.cast Nx.float32 w64

(* A captured tensor of [x]'s dtype. *)
let captured (type b) (x : (float, b) Nx.t) : (float, b) Nx.t =
  match Nx.dtype x with
  | Float32 -> w32
  | Float64 -> w64
  | dt -> Format.kasprintf invalid_arg "no capture of %a" Nx.pp_dtype dt

(* Its backward pass reads values the forward computes (a product, a tanh and an
   exponential), the argument and a movement of it, movements of a capture, and
   a selection against a constant. *)
let mixed =
  let f x =
    let w =
      Nx.reshape (Nx.shape x) (Nx.slice [ R (0, Nx.numel x) ] (captured x))
    in
    let computed = Nx.mul (Nx.tanh (Nx.mul x x)) (Nx.exp x) in
    let moved = Nx.mul (Nx.mul (Nx.flip x) x) w in
    let zero = Nx.zeros_like x in
    let selected = Nx.where (Nx.greater x zero) (Nx.mul x x) zero in
    Nx.add (Nx.add computed moved) selected
  in
  { name = "computed values, arguments, captures and constants"; f }

let remat =
  let f x =
    let inner =
      Rune.remat
        Nx.Ptree.(tensor @-> returns tensor)
        (fun y -> Nx.tanh (Nx.mul y y))
    in
    Nx.mul (inner (Nx.mul_s x 2.)) x
  in
  { name = "a remat"; f }

(* A scan over [x]'s rows, whose outputs are the carries. *)
let scan =
  let f x =
    let k = (Nx.shape x).(1) in
    snd
      (Rune.scan'
         ~f:(fun c row ->
           let c = Nx.add (Nx.mul c (Nx.tanh row)) (Nx.sin row) in
           (c, c))
         ~init:(Nx.ones (Nx.dtype x) [| k |])
         x)
  in
  { name = "a scan"; f }

(* One compiled function per dtype, shared by every case of every law, so that
   each key traces once. *)
type compiled = {
  f32 : Nx.float32_t -> Nx.float32_t;
  f64 : Nx.float64_t -> Nx.float64_t;
}

let compile fn = { f32 = Rune.jit' fn.f; f64 = Rune.jit' fn.f }

let at (type b) c (dt : (float, b) Nx.dtype) :
    (float, b) Nx.t -> (float, b) Nx.t =
  match dt with
  | Float32 -> c.f32
  | Float64 -> c.f64
  | dt -> Format.kasprintf invalid_arg "no function at %a" Nx.pp_dtype dt

(* Law 3: jit is the identity under every transformation *)

type transformation = {
  t_name : string;
  apply :
    'b.
    ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t list;
  rows : bool;  (** Whether it maps over the argument's rows. *)
}

let loss f x = Nx.sum (f x)
let total : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let transformations =
  [
    {
      t_name = "grad";
      apply = (fun f x -> [ Rune.grad' (loss f) x ]);
      rows = false;
    };
    {
      t_name = "vjp";
      apply =
        (fun f x ->
          let y, pullback = Rune.vjp' f x in
          [ y; pullback (Nx.cos y) ]);
      rows = false;
    };
    {
      t_name = "jvp";
      apply =
        (fun f x ->
          let y, dy = Rune.jvp' f x (Nx.sin x) in
          [ y; dy ]);
      rows = false;
    };
    { t_name = "vmap"; apply = (fun f x -> [ Rune.vmap' f x ]); rows = true };
    {
      t_name = "a total around grad";
      apply =
        (fun f x ->
          let counted x =
            Rune.Total.add total (Nx.cast Nx.float64 (Nx.sum x));
            f x
          in
          let g, sum =
            Rune.Total.collect total ~zero:(Nx.scalar Nx.float64 0.) (fun () ->
                Rune.grad' (loss counted) x)
          in
          [ g; Nx.cast (Nx.dtype x) sum ]);
      rows = false;
    };
    {
      t_name = "vmap of grad";
      apply = (fun f x -> [ Rune.vmap' (Rune.grad' (loss f)) x ]);
      rows = true;
    };
    {
      t_name = "jvp of grad";
      apply =
        (fun f x -> [ snd (Rune.jvp' (Rune.grad' (loss f)) x (Nx.sin x)) ]);
      rows = false;
    };
    {
      t_name = "grad of grad";
      apply = (fun f x -> [ Rune.grad' (loss (Rune.grad' (loss f))) x ]);
      rows = false;
    };
  ]

(* The functions with the shapes they take, alone and under a map. *)
let functions =
  [
    (mixed, shapes, matrices);
    (remat, shapes, matrices);
    (scan, rows, lanes_of_rows);
  ]

let law (fn, shapes, mapped) c t =
  let shapes = if t.rows then mapped else shapes in
  prop ~count:12 ~examples:(empty shapes)
    (Printf.sprintf "%s (jit f) is %s f" t.t_name t.t_name) (input shapes)
    (fun (Input (dt, x)) ->
      cover "no element" (Nx.numel x = 0);
      cover "float32" (Nx_dtype.equal dt Nx.float32);
      cover "float64" (Nx_dtype.equal dt Nx.float64);
      let expected = t.apply fn.f x and actual = t.apply (at c dt) x in
      List.iteri
        (fun i e ->
          equal ~msg:(string_of_int i) (close dt) e (List.nth actual i))
        expected)

let identity =
  group "jit is the identity under a transformation"
    (List.map
       (fun ((fn, _, _) as f) ->
         let c = compile fn in
         group fn.name (List.map (law f c) transformations))
       functions)

(* A compiled function that consumes its argument consumes nothing under a
   transformation, so the argument stays readable for the backward pass. *)
let consumed =
  prop ~count:12 ~examples:(empty shapes)
    "grad of a consuming compiled function is grad f" (input shapes)
    (fun (Input (dt, x)) ->
      let g = Rune.jit Nx.Ptree.(consumes tensor @@ returns tensor) mixed.f in
      let before = Nx.copy x in
      equal (close dt) (Rune.grad' (loss mixed.f) x) (Rune.grad' (loss g) x);
      equal ~msg:"the argument" (close dt) before x)

(* Law 4: one forward *)

(* [profiled f] is [f ()] and the names of the compiled call's host spans
   meanwhile. *)
let profiled f =
  let p = Nx_device.Profile.start () in
  match f () with
  | y ->
      let spans =
        List.filter_map
          (function
            | Nx_device.Profile.Span s
              when String.starts_with ~prefix:"rune.jit: " s.name ->
                Some s.name
            | _ -> None)
          (Nx_device.Profile.stop p)
      in
      (y, spans)
  | exception e ->
      ignore (Nx_device.Profile.stop p);
      raise e

(* The residual traces and the other traces while [f ()] runs. *)
let traces f =
  let y, spans = profiled f in
  let count name = List.length (List.filter (String.equal name) spans) in
  (y, (count "rune.jit: residuals", count "rune.jit: trace"))

let x () = values Nx.float32 [| 2; 3 |] 1
let w () = values Nx.float32 [| 2; 3 |] 2
let pair = pair int int

let one_forward =
  group "one forward"
    [
      test
        "grad of a compiled function traces its residuals, its forward and its \
         backward once per key" (fun () ->
          let g = Rune.jit' mixed.f in
          let _, first = traces (fun () -> Rune.grad' (loss g) (x ())) in
          equal ~msg:"first call" pair (1, 2) first;
          let _, again = traces (fun () -> Rune.grad' (loss g) (w ())) in
          equal ~msg:"same key" pair (0, 0) again;
          let other = values Nx.float32 [| 4 |] 3 in
          let _, shape = traces (fun () -> Rune.grad' (loss g) other) in
          equal ~msg:"another shape" pair (1, 2) shape);
      test "another set of tracked arguments traces its residuals again"
        (fun () ->
          let g =
            Rune.jit
              Nx.Ptree.(tensor @-> tensor @-> returns tensor)
              (fun x w -> Nx.mul (Nx.tanh x) (Nx.exp w))
          in
          let of_x () = Rune.grad' (fun x -> Nx.sum (g x (w ()))) (x ()) in
          let of_both () =
            Rune.grad
              Nx.Ptree.(pair tensor tensor)
              (fun (x, w) -> Nx.sum (g x w))
              (x (), w ())
          in
          equal ~msg:"x" pair (1, 2) (snd (traces of_x));
          equal ~msg:"x and w" pair (1, 2) (snd (traces of_both));
          equal ~msg:"x again" pair (0, 0) (snd (traces of_x));
          equal ~msg:"x and w again" pair (0, 0) (snd (traces of_both)));
      test
        "the backward pass traces once per layout of the cotangents and never \
         runs the function" (fun () ->
          let runs = ref 0 in
          let g =
            Rune.jit' (fun x ->
                incr runs;
                mixed.f x)
          in
          let (y, pullback), first = traces (fun () -> Rune.vjp' g (x ())) in
          equal ~msg:"the forward call" pair (1, 1) first;
          equal ~msg:"runs of the function by the forward call" int 2 !runs;
          let dense () = Nx.cos y in
          equal ~msg:"a dense cotangent" pair (0, 1)
            (snd (traces (fun () -> pullback (dense ()))));
          equal ~msg:"another dense cotangent" pair (0, 0)
            (snd (traces (fun () -> pullback (Nx.sin y))));
          let broadcast =
            Nx.broadcast_to [| 2; 3 |] (Nx.scalar Nx.float32 1.)
          in
          let g', spans = traces (fun () -> pullback broadcast) in
          equal ~msg:"a broadcast cotangent" pair (0, 1) spans;
          equal ~msg:"runs of the function by the pullbacks" int 2 !runs;
          equal ~msg:"the broadcast cotangent's gradient" (close Nx.float32)
            (snd (Rune.vjp' mixed.f (x ())) broadcast)
            g');
      test
        "the forward pass runs once per call: a total counts it once, and the \
         function runs only in the call's two traces" (fun () ->
          let t = Rune.Total.make () and runs = ref 0 in
          let g =
            Rune.jit' (fun x ->
                incr runs;
                Rune.Total.add t (Nx.ones Nx.float32 [||]);
                mixed.f x)
          in
          let (), total =
            Rune.Total.collect t ~zero:(Nx.zeros Nx.float32 [||]) (fun () ->
                for _ = 1 to 3 do
                  ignore (Rune.grad' (loss g) (x ()))
                done)
          in
          equal ~msg:"the total" (Oracle.tensor ()) (Nx.scalar Nx.float32 3.)
            total;
          equal ~msg:"runs of the function" int 2 !runs);
      test
        "grad of a compiled function that calls another is grad f, and neither \
         runs in the pullback" (fun () ->
          let outer = ref 0 and inner = ref 0 in
          let h =
            Rune.jit' (fun x ->
                incr inner;
                Nx.mul (Nx.tanh x) (Nx.exp x))
          in
          let g =
            Rune.jit' (fun x ->
                incr outer;
                Nx.sin (h (Nx.mul x x)))
          in
          let f x =
            Nx.sin (Nx.mul (Nx.tanh (Nx.mul x x)) (Nx.exp (Nx.mul x x)))
          in
          let y, pullback = Rune.vjp' g (x ()) in
          let runs = (!outer, !inner) in
          let ct = Nx.cos y in
          let expected_y, expected = Rune.vjp' f (x ()) in
          equal ~msg:"the result" (close Nx.float32) expected_y y;
          equal ~msg:"the gradient" (close Nx.float32) (expected ct)
            (pullback ct);
          equal ~msg:"runs in the pullback" pair runs (!outer, !inner));
      test
        "a transpose that cannot be traced raises Jit_error at the forward call"
        (fun () ->
          let f =
            Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
                ( Nx.sin x,
                  fun ct -> Nx.mul_s ct (Nx.item [] (Nx.sum (Nx.cos x))) ))
          in
          ignore (Rune.vjp' f (x ()));
          raises_match
            (function Rune.Jit_error _ -> true | _ -> false)
            (fun () -> ignore (Rune.vjp' (Rune.jit' f) (x ()))));
    ]

(* Residuals *)

(* A device over the host's memory whose allocations the tests count. *)
let device = Nx.Device.v (Cpu 3)
let on_device x = Nx.place (Nx.Placement.on device) x

let settled () =
  let m = Nx.Device.memory device in
  for _ = 1 to 4 do
    Gc.full_major ();
    Nx_device.synchronize m
  done;
  Nx_device.Stats.allocated (Nx_device.stats m)

(* [held f] is the bytes [f ()]'s result holds on the device. *)
let held f =
  let before = settled () in
  let r = f () in
  let held = settled () - before in
  ignore (Sys.opaque_identity r);
  held

let residuals =
  group "residuals"
    [
      test
        "a transpose that reads an argument, its movements and a capture adds \
         no result to the forward call" (fun () ->
          let w = on_device (values Nx.float32 [| 32; 64 |] 5) in
          let g =
            Rune.jit' (fun x ->
                let t = Nx.transpose x in
                Nx.mul (Nx.mul t t) w)
          in
          let x = on_device (values Nx.float32 [| 64; 32 |] 6) in
          let call () = g x and forward () = Rune.vjp' g x in
          (* The first runs load the programs, and settle what earlier tests
             left. *)
          ignore (held call);
          ignore (held forward);
          equal ~msg:"bytes held" int (held call) (held forward));
      test
        "a function that computes other values on another call raises \
         Jit_error naming determinism" (fun () ->
          let runs = ref 0 in
          let g =
            Rune.jit' (fun x ->
                incr runs;
                if !runs = 1 then Nx.sin x else Nx.exp (Nx.sin x))
          in
          raises
            (Rune.Jit_error
               "Rune.jit: the function computed other values on this call than \
                when its reverse mode was split: a compiled function must be \
                deterministic, computing the same operations whenever its \
                arguments have the same dtypes, shapes and placements")
            (fun () -> ignore (Rune.grad' (loss g) (x ()))));
      test
        "grad through a custom_jvp whose tangent map calls a compiled function \
         is the rule's derivative" (fun () ->
          let scale =
            Rune.jit
              Nx.Ptree.(tensor @-> tensor @-> returns tensor)
              (fun c d -> Nx.mul c d)
          in
          let f =
            Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
                (Nx.sin x, fun dx -> scale (Nx.cos x) dx))
          in
          equal (close Nx.float32)
            (Nx.cos (x ()))
            (Rune.grad' (fun x -> Nx.sum (f x)) (x ())));
    ]

(* Map collectives *)

(* A function of a lane's row that reads every lane's row, the lane's index and
   a mean over the lanes, whose transpose reads all three. *)
let collective a x =
  let mean = Nx.mean ~axes:[ 0 ] (Rune.lanes a x) in
  let index = Nx.cast (Nx.dtype x) (Rune.lane_index ~axis:a ()) in
  Nx.sum (Nx.mul (Nx.mul x index) (Nx.tanh mean))

let collectives =
  let a = Rune.axis () in
  let g = Rune.jit' (collective a) in
  prop ~count:8 "vmap (grad (jit f)) is vmap (grad f) at two lane counts"
    Gen.(pair (of_list ~pp:Nx.pp_shape [ [| 3 |]; [| 0 |] ]) (int_range 0 1000))
    (fun (row, seed) ->
      cover "no element" (Array.fold_left ( * ) 1 row = 0);
      List.iter
        (fun n ->
          let x = values Nx.float32 (Array.append [| n |] row) seed in
          let map f = Rune.vmap' ~axis:a (Rune.grad' f) x in
          equal
            ~msg:(Printf.sprintf "%d lanes" n)
            (close Nx.float32)
            (map (collective a))
            (map g))
        [ 4; 7 ])

(* Refusals *)

let tracked =
  "Rune.jit: the function reads, through its closure, a value a transformation \
   tracks (float32[2,3] on CPU). Pass it as an argument."

let refusals =
  group "refusals"
    [
      test "under grad, a value grad tracks read through the closure raises"
        (fun () ->
          raises (Invalid_argument tracked) (fun () ->
              ignore
                (Rune.grad'
                   (fun w -> Nx.sum (Rune.jit' (fun a -> Nx.mul a w) (x ())))
                   (w ()))));
      test "under jvp, a value jvp tracks read through the closure raises"
        (fun () ->
          raises (Invalid_argument tracked) (fun () ->
              ignore
                (Rune.jvp'
                   (fun w -> Rune.jit' (fun a -> Nx.mul a w) (x ()))
                   (w ()) (x ()))));
      test "under vmap, a lane read through the closure raises" (fun () ->
          let rows = values Nx.float32 [| 4; 2; 3 |] 4 in
          raises (Invalid_argument tracked) (fun () ->
              ignore
                (Rune.vmap'
                   (fun w -> Rune.jit' (fun a -> Nx.mul a w) (x ()))
                   rows)));
      test "a traced value that escaped its trace raises at a later compile"
        (fun () ->
          let leaked = ref None in
          let f =
            Rune.jit' (fun x ->
                leaked := Some (Nx.exp x);
                x)
          in
          ignore (f (x ()));
          let g = Rune.jit' (fun y -> Nx.add y (Option.get !leaked)) in
          raises
            (Invalid_argument
               "Rune.jit: a traced value escaped the function that traced it; \
                return it from that function instead") (fun () ->
              ignore (g (w ()))));
    ]

let () =
  exit
    (run "Rune transformations of jit"
       [
         identity;
         group "consumption" [ consumed ];
         one_forward;
         residuals;
         group "map collectives" [ collectives ];
         refusals;
       ])
