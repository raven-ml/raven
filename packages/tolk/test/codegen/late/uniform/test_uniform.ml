(* Tests of Tolk.Uniform: which values may differ between a workgroup's threads,
   and a kernel whose barrier or matrix multiply-accumulate runs in a loop whose
   trips may differ, refused where it is lowered. Kernels that may diverge are
   lowered, never run. *)

open Windtrap
open Tolk
module O = Ops.O

let gs = 4 (* workgroups *)
let ls = 32 (* threads of a workgroup *)
let i32 n = Ops.int ~dtype:Int32 n
let int = O.int
let ld b i = Ops.load (Ops.index b [ i ]) []
let st b i v = Ops.store (Ops.index b [ i ]) v
let buffer ?volatile slot n dt = Shape.param ~shape:[ Int n ] ?volatile slot dt
let local slot n dt = Shape.placeholder ~slot ~addrspace:Local [ n ] dt
let reg slot n dt = Shape.placeholder ~slot ~addrspace:Reg [ n ] dt
let range ty bound id = Ops.range ~axis_type:ty bound [ id ]
let workgroup () = range Global (Int gs) 0
let thread () = range Local (Int ls) 1
let at g l = O.((g * int ls) + l)
let weak u = Ops.cast u Weak_int
let n = Ops.variable "n" (Dtype.Value.of_int 0) (Dtype.Value.of_int 8)

let kernel name effects =
  Ops.sink ~kernel:(Ops.kernel_info ~name ~opts_to_apply:[] ()) effects

(* HIP's renderer, which lowering needs and compiling would not: making it finds
   comgr, which the suite's COMGR_PATH stands in for. *)
let hip =
  lazy
    (Renderer.with_compiler
       (Renderer.Compiler.v Fun.id)
       (Cstyle.hip
          {
            Helpers.Target.device = "AMD";
            renderer = "HIP";
            arch = "gfx1100";
            interface = "";
            indices = "";
          }))

let lowered k = Codegen.full_rewrite_to_sink k (Lazy.force hip)

(* Values *)

(* A kernel's values, each stored so that the kernel holds it: [x] is read only,
   [y] written, [v] volatile. *)
let g = workgroup ()
let l = thread ()
let x = buffer 2 (gs * ls) Int32
let y = buffer 3 (gs * ls) Int32
let v = buffer ~volatile:true 4 1 Int32
let serial bound id = range Loop (Sym bound) id
let wrote_y = st y (at g l) (i32 0)

(* [acc] summed over [trips] trips, each adding [add], read after the loop. *)
let carried slot trips add =
  let acc = reg slot 1 Int32 in
  let init = st acc (int 0) (i32 0) in
  let j = serial trips (20 + slot) in
  let acc_j = Ops.after acc [ init; j ] in
  let sum = Ops.end_ (st acc_j (int 0) O.(ld acc_j (int 0) + add)) [ j ] in
  ld (Ops.after acc [ sum ]) (int 0)

let uniform_sum = carried 30 (weak (ld x g)) (ld x g)
let lane_sum = carried 31 (int 4) (ld x (at g l))
let tripped_sum = carried 32 O.(l + int 1) (i32 1)
let fed_sum = carried 33 (int 4) lane_sum

(* Workgroup memory [b] written at each thread's slot, then read at slot 0. *)
let staged_read slot ~barrier =
  let b = local slot ls Int32 in
  let w = st b l (ld x (at g l)) in
  let w = if barrier then Ops.barrier w [] else w in
  (b, ld (Ops.after b [ w ]) (int 0))

let read_after_barrier = snd (staged_read 40 ~barrier:true)
let read_without_barrier = snd (staged_read 41 ~barrier:false)

(* The same read, with each thread's slot written again after it, with no
   barrier between. *)
let read_then_written, written_again =
  let b, read = staged_read 42 ~barrier:true in
  (read, st (Ops.after b [ read ]) l (i32 1))

let value_cases =
  [
    ("a constant", i32 3, false);
    ("a variable", n, false);
    ("a workgroup index", g, false);
    ("a thread index", l, true);
    ("a warp's lane", range Warp (Int 32) 5, true);
    ("a vector lane", range Upcast (Int 4) 6, true);
    ("a workgroup index named gidx", Ops.special (Int gs) "gidx0", false);
    ("a thread index named lidx", Ops.special (Int ls) "lidx0", true);
    ("a loop's index of a uniform bound", serial (weak (ld x g)) 7, false);
    ("a loop's index of a varying bound", serial O.(l + int 1) 8, true);
    ("arithmetic of uniform values", O.((g * int ls) + n), false);
    ("arithmetic reading a thread index", at g l, true);
    ("a load of a read-only buffer at a uniform index", ld x g, false);
    ("a load at a thread index", ld x (at g l), true);
    ( "a load under a gate reading a thread index",
      Ops.load (Ops.index x [ Ops.where O.(l < int 3) g Ops.invalid ]) [],
      true );
    ("a load of a buffer the kernel writes", ld y g, true);
    ("a load of a volatile parameter", ld v (int 0), true);
    ( "an after of a uniform value, ordered after a thread's store",
      Ops.after (ld x g) [ wrote_y ],
      false );
    ( "a maximum over the thread index",
      Ops.reduce (ld x (at g l)) Max [ l ],
      false );
    ( "a sum over a loop of a varying bound",
      Ops.reduce (ld x g) Add [ serial O.(l + int 1) 9 ],
      true );
    ( "a sum over a loop of a uniform bound",
      Ops.reduce (ld x g) Add [ range Loop (Int 4) 10 ],
      false );
    ("a register carrying uniform values", uniform_sum, false);
    ("a register carrying a thread's values", lane_sum, true);
    ("a register carried through a loop of varying trips", tripped_sum, true);
    ("a register fed from a varying register", fed_sum, true);
    ("workgroup memory read after a barrier", read_after_barrier, false);
    ("workgroup memory read without a barrier", read_without_barrier, true);
    ("workgroup memory written again after it is read", read_then_written, true);
  ]

let values_kernel =
  let out = buffer 0 (List.length value_cases) Int32 in
  kernel "values"
    (wrote_y :: written_again
    :: List.mapi
         (fun k (_, u, _) -> st out (int k) (Ops.cast u Int32))
         value_cases)

let custom_values =
  let custom = Ops.v Op.Customi ~arg:(Code { code = "f()"; dtype = Int32 }) in
  let ins = Ops.v Op.Ins ~arg:(Code { code = "s_nop"; dtype = Int32 }) in
  [ ("custom code", custom); ("an instruction", ins) ]

let values =
  group "varying"
    [
      cases "decides from what a value reads"
        ~id:(fun (s, _, _) -> s)
        value_cases
        (fun (_, u, expected) ->
          equal bool expected (Uniform.varying values_kernel u));
      cases "custom code and instructions vary" ~id:fst custom_values
        (fun (_, u) ->
          let k = kernel "custom" [ st y g u ] in
          equal bool true (Uniform.varying k u));
      cases "a body another kernel calls takes as varying its" ~id:fst
        [ ("parameter", Shape.param 0 Int32); ("variable", n) ]
        (fun (_, p) ->
          let body = Ops.sink [ st y g (Ops.cast p Int32) ] in
          let launched = kernel "launched" [ st y g (Ops.cast p Int32) ] in
          equal bool ~msg:"in a body" true (Uniform.varying body p);
          equal bool ~msg:"in a launched kernel" false
            (Uniform.varying launched p));
    ]

(* Lowered kernels *)

(* [out[g*ls + l]] is [inp] at thread [l]'s neighbour, staged through workgroup
   memory in a loop of [bound] trips: each trip, each thread writes its slot, a
   barrier waits, and each reads its neighbour's. *)
let staged_loop ?(writes = []) name bound =
  let out = buffer 0 (gs * ls) Float32 and inp = buffer 1 (gs * ls) Float32 in
  let t = range Loop (Sym bound) 2 in
  let b = local 10 ls Float32 in
  let w = st b l (ld (Ops.after inp [ t ]) (at g l)) in
  let read = ld (Ops.after b [ w ]) O.((l + int 1) % int ls) in
  let loop = Ops.end_ (st out (at g l) read) [ t ] in
  kernel name [ Ops.end_ (Ops.group (loop :: writes)) [ g; l ] ]

(* The thread [l] of workgroup [g] writes [x], then the loop runs as many trips
   as slot 0 of workgroup memory holds after a barrier. *)
let after_barrier () =
  let b = local 12 ls Int32 in
  let w = st b l (ld x (at g l)) in
  weak (ld (Ops.after b [ w ]) (int 0))

(* A register counting the trips of a loop of [l + 1] trips. *)
let counted () =
  let acc = reg 11 1 Int32 in
  let init = st acc (int 0) (i32 0) in
  let j = range Loop (Sym O.(l + int 1)) 3 in
  let acc_j = Ops.after acc [ init; j ] in
  let count = Ops.end_ (st acc_j (int 0) O.(ld acc_j (int 0) + i32 1)) [ j ] in
  weak (ld (Ops.after acc [ count ]) (int 0))

let vote () = weak (Ops.reduce (ld x (at g l)) Max [ l ])

let refused =
  [
    ("lidx + 1", fun () -> staged_loop "plus_one" O.(l + int 1));
    ("x[lidx]", fun () -> staged_loop "lane_load" (weak (ld x l)));
    ( "x[gidx], x written",
      fun () ->
        staged_loop ~writes:[ st x (at g l) (i32 0) ] "written" (weak (ld x g))
    );
    ("a volatile load", fun () -> staged_loop "volatile" (weak (ld v (int 0))));
    ( "a register counting a loop of lidx + 1 trips",
      fun () -> staged_loop "counted" (counted ()) );
  ]

let accepted =
  [
    ("x[gidx], x read only", fun () -> staged_loop "read_only" (weak (ld x g)));
    ("a variable", fun () -> staged_loop "variable" n);
    ("the maximum of x over the threads", fun () -> staged_loop "vote" (vote ()));
    ( "workgroup memory read after its barrier",
      fun () -> staged_loop "after_barrier" (after_barrier ()) );
  ]

let flags = buffer 3 32 Int32

(* Per workgroup, one warp computes C (16x8 f32) = A (16x16 bf16) B (16x8) with
   one matrix multiply-accumulate, B given transposed, in a loop of [bound lane]
   trips. *)
let product_loop name bound =
  let out = buffer 0 (gs * 128) Float32
  and a = buffer 1 (gs * 256) Bfloat16
  and bt = buffer 2 (gs * 128) Bfloat16 in
  let lane = range Warp (Int 32) 1 in
  let s = range Loop (Sym (bound lane)) 3 in
  let a_s = Ops.after a [ s ] and bt_s = Ops.after bt [ s ] in
  let gq = O.(lane // int 4) and tq = O.(lane % int 4) in
  let c0 = O.(int 2 * tq) and g8 = O.(gq + int 8) in
  let am r c = ld a_s O.((g * int 256) + (r * int 16) + c) in
  let bm c k = ld bt_s O.((g * int 128) + (c * int 16) + k) in
  let afrag =
    Shape.stack
      [
        am gq c0;
        am gq O.(c0 + int 1);
        am g8 c0;
        am g8 O.(c0 + int 1);
        am gq O.(c0 + int 8);
        am gq O.(c0 + int 9);
        am g8 O.(c0 + int 8);
        am g8 O.(c0 + int 9);
      ]
  in
  let bfrag =
    Shape.stack
      [
        bm gq c0;
        bm gq O.(c0 + int 1);
        bm gq O.(c0 + int 8);
        bm gq O.(c0 + int 9);
      ]
  in
  let zero = Ops.float ~dtype:Float32 0. in
  let acc = Shape.stack [ zero; zero; zero; zero ] in
  let d = Ops.wmma afrag bfrag ~acc ~dims:(8, 16, 16) ~threads:32 in
  let elt e = Ops.index d [ i32 e ] in
  let to_out r c = O.((g * int 128) + (r * int 8) + c) in
  let stores =
    [
      st out (to_out gq c0) (elt 0);
      st out (to_out gq O.(c0 + int 1)) (elt 1);
      st out (to_out g8 c0) (elt 2);
      st out (to_out g8 O.(c0 + int 1)) (elt 3);
    ]
  in
  kernel name [ Ops.end_ (Ops.end_ (Ops.group stores) [ s ]) [ g; lane ] ]

let inp_of_loop = buffer 1 (gs * ls) Float32

let checks =
  group "check"
    [
      cases "refuses a barrier in a loop whose bound is" ~id:fst refused
        (fun (_, k) ->
          raises_match (Exn.invalid_arg ~sub:"a barrier runs in loop")
            (fun () -> lowered (k ())));
      cases "accepts a barrier in a loop whose bound is" ~id:fst accepted
        (fun (_, k) -> ignore (lowered (k ())));
      test "the refusal names the kernel, the operation and the loop" (fun () ->
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "kernel plus_one: a barrier runs in loop 2, whose trips may \
                  differ between the threads of a workgroup") (fun () ->
              lowered (staged_loop "plus_one" O.(l + int 1))));
      test "refuses a matrix multiply-accumulate in a loop bounded by x[lane]"
        (fun () ->
          let k =
            product_loop "lane_product" (fun lane -> weak (ld flags lane))
          in
          raises_match
            (Exn.invalid_arg
               ~sub:"kernel lane_product: a matrix multiply-accumulate runs")
            (fun () -> lowered k));
      test "accepts a matrix multiply-accumulate in a loop bounded by x[gidx]"
        (fun () ->
          ignore
            (lowered
               (product_loop "workgroup_product" (fun _ -> weak (ld flags g)))));
      test "checks the kernel it is given, after its barriers are placed"
        (fun () ->
          let t = range Loop (Sym O.(l + int 1)) 2 in
          let b = local 10 ls Float32 in
          let w = st b l (ld (Ops.after x [ t ]) l) in
          let k =
            kernel "placed"
              [ Ops.end_ (Ops.end_ (Ops.barrier w []) [ t ]) [ g; l ] ]
          in
          raises_match (Exn.invalid_arg ~sub:"kernel placed: a barrier")
            (fun () -> Uniform.check k));
      cases "an unbounded loop holding a barrier, repeated while"
        ~id:(fun (s, _, _) -> s)
        [
          ("a thread's flag holds, is refused", O.(ld x (at g l) <> i32 0), true);
          ("its workgroup's flag holds, is accepted", O.(ld x g <> i32 0), false);
        ]
        (fun (_, cond, refused) ->
          let r = Ops.range ~dtype:Void (Sym (Ops.v Op.Noop)) [ 2 ] in
          let b = local 10 ls Float32 in
          let w = st b l (ld (Ops.after inp_of_loop [ r ]) l) in
          let loop = Ops.backedge (Ops.barrier w []) ~loop:r ~cond in
          let k = kernel "repeated" [ Ops.end_ loop [ g; l ] ] in
          if refused then
            raises_match (Exn.invalid_arg ~sub:"kernel repeated: a barrier")
              (fun () -> Uniform.check k)
          else Uniform.check k);
      test "names a called body, which has no name, in its refusal" (fun () ->
          let t = range Loop (Sym O.(l + int 1)) 2 in
          let b = local 10 ls Float32 in
          let w = st b l (ld (Ops.after x [ t ]) l) in
          let body = Ops.sink [ Ops.end_ (Ops.barrier w []) [ t ] ] in
          raises_match (Exn.invalid_arg ~sub:"a called body: a barrier")
            (fun () -> Uniform.check body));
    ]

let () = exit (run "Tolk.Uniform" [ values; checks ])
