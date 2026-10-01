module Rune = Rune_next.Rune
let f32 = Nx.float32
let d = match Metal.device with Some d -> d | None -> failwith "no metal"
let on = Nx.Placement.device (Nx.Device.of_runtime d)
let at x = Nx.place on x
let host x = Nx.place Nx.Placement.host x
let ran = ref 0

let rows n k = Nx.reshape [| n; k |] (Nx.mul_s (Nx.arange_f f32 0. (Float.of_int (n * k)) 1.) 0.01)

let maxdiff a b = Nx.item [] (Nx.max (Nx.abs (Nx.sub (Nx.cast Nx.float64 a) (Nx.cast Nx.float64 b))))

let check name ?(tol=1e-4) eager staged =
  match eager () with
  | exception e -> Printf.printf "%-50s eager raised %s\n%!" name (Printexc.to_string e)
  | e ->
    ran := 0;
    match staged () with
    | exception e -> Printf.printf "%-50s RAISE %s\n%!" name (Printexc.to_string e)
    | s ->
      let dd = List.fold_left2 (fun m a b -> Float.max m (maxdiff a (host b))) 0. e s in
      Printf.printf "%-50s %s diff=%.3g steps=%d\n%!" name (if dd <= tol then "OK  " else "FAIL") dd !ran

let lst = Nx.Ptree.(tensor @-> returns (list tensor))
let scan2 f init xs = let c, ys = Rune.scan' ~f:(fun c x -> incr ran; f c x) ~init xs in [c; ys]

let case name ?tol init f xs =
  check name ?tol (fun () -> scan2 f init xs) (fun () -> Rune.jit lst (fun xs -> scan2 f init xs) (at xs))

let () =
  (* B: swap two carries (one pair carry in a structure) *)
  let swap xs =
    let (a, b), ys = Rune.scan Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor Nx.Ptree.tensor
      ~f:(fun (a, b) x -> incr ran; ((Nx.add b x, a), Nx.add a b)) ~init:(Nx.ones f32 [|4|], Nx.full f32 [|4|] 3.) xs in [a; b; ys] in
  check "swap two carries" (fun () -> swap (rows 6 4)) (fun () -> Rune.jit lst swap (at (rows 6 4)));
  let swap_pure xs =
    let (a, b), ys = Rune.scan Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor Nx.Ptree.tensor
      ~f:(fun (a, b) x -> incr ran; ((b, a), Nx.add (Nx.mul_s a 2.) x)) ~init:(Nx.ones f32 [|4|], Nx.full f32 [|4|] 3.) xs in [a; b; ys] in
  check "swap two carries, pure" (fun () -> swap_pure (rows 5 4)) (fun () -> Rune.jit lst swap_pure (at (rows 5 4)));
  let fib xs =
    let (a, b), ys = Rune.scan Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor Nx.Ptree.tensor
      ~f:(fun (a, b) x -> incr ran; ((b, Nx.add (Nx.add a b) x), a)) ~init:(Nx.ones f32 [|4|], Nx.ones f32 [|4|]) xs in [a; b; ys] in
  check "fibonacci carries" (fun () -> fib (rows 6 4)) (fun () -> Rune.jit lst fib (at (rows 6 4)));
  (* C: output the old carry *)
  case "output old carry, next flipped" (Nx.ones f32 [|4|]) (fun c x -> (Nx.add (Nx.flip c) x, c)) (rows 6 4);
  case "output old carry, next own" (Nx.ones f32 [|4|]) (fun c x -> (Nx.add c x, c)) (rows 6 4);
  case "output old carry rolled" (Nx.ones f32 [|4|]) (fun c x -> (Nx.add c x, Nx.flip c)) (rows 6 4);
  (* D: host capture in step *)
  let wh = Nx.create f32 [|4|] [|0.5;-0.25;1.5;2.|] in
  case "host capture in step" (Nx.ones f32 [|4|]) (fun c x -> let c = Nx.add (Nx.mul c wh) x in (c, c)) (rows 6 4);
  (* E: metal capture *)
  let wm = at wh in
  check "metal capture in step"
    (fun () -> scan2 (fun c x -> let c = Nx.add (Nx.mul c wh) x in (c, c)) (Nx.ones f32 [|4|]) (rows 6 4))
    (fun () -> Rune.jit lst (fun xs -> scan2 (fun c x -> let c = Nx.add (Nx.mul c wm) x in (c, c)) (Nx.ones f32 [|4|]) xs) (at (rows 6 4)));
  check "capture computed in the function before the scan"
    (fun () -> let xs = rows 6 4 in let w = Nx.exp (Nx.sum ~axes:[0] xs) in scan2 (fun c x -> let c = Nx.add (Nx.mul c w) x in (c, Nx.mul c w)) (Nx.ones f32 [|4|]) xs)
    (fun () -> Rune.jit lst (fun xs -> let w = Nx.exp (Nx.sum ~axes:[0] xs) in scan2 (fun c x -> let c = Nx.add (Nx.mul c w) x in (c, Nx.mul c w)) (Nx.ones f32 [|4|]) xs) (at (rows 6 4)));
  (* J/K: constant output, row output *)
  case "constant output" (Nx.ones f32 [|4|]) (fun c x -> (Nx.add c x, Nx.ones f32 [|3|])) (rows 6 4);
  case "row output" (Nx.ones f32 [|4|]) (fun c x -> (Nx.add c x, x)) (rows 6 4);
  case "unchanged carry" (Nx.ones f32 [|4|]) (fun c x -> (c, Nx.add c x)) (rows 6 4);
  case "scalar output" (Nx.ones f32 [|3|]) (fun c x -> (Nx.add c x, Nx.sum c)) (rows 7 3);
  case "scalar carry" (Nx.scalar f32 1.) (fun c x -> (Nx.add c (Nx.sum x), Nx.mul_s x 2.)) (rows 7 3);
  case "one step" (Nx.ones f32 [|3|]) (fun c x -> (Nx.add (Nx.flip c) x, c)) (rows 1 3);
  (* G: carry shape change -> declined *)
  check "carry changes shape (declined)"
    (fun () -> [fst (Rune.scan' ~f:(fun c x -> incr ran; (Nx.concatenate ~axis:0 [c; Nx.slice [I 0] x |> Nx.reshape [|1|]], Nx.sum x)) ~init:(Nx.ones f32 [|1|]) (rows 4 3))])
    (fun () -> Rune.jit Nx.Ptree.(tensor @-> returns (list tensor)) (fun xs -> [fst (Rune.scan' ~f:(fun c x -> incr ran; (Nx.concatenate ~axis:0 [c; Nx.slice [I 0] x |> Nx.reshape [|1|]], Nx.sum x)) ~init:(Nx.ones f32 [|1|]) xs)]) (at (rows 4 3)));
  (* U: non-contiguous stacked input *)
  check "transposed stacked input"
    (fun () -> scan2 (fun c x -> let c = Nx.add c x in (c, c)) (Nx.zeros f32 [|5|]) (Nx.transpose (rows 5 6)))
    (fun () -> Rune.jit lst (fun xs -> scan2 (fun c x -> let c = Nx.add c x in (c, c)) (Nx.zeros f32 [|5|]) (Nx.transpose xs)) (at (rows 5 6)));
  check "sliced stacked input (offset view)"
    (fun () -> scan2 (fun c x -> let c = Nx.add c x in (c, c)) (Nx.zeros f32 [|3|]) (Nx.slice [R (1, 6); R (1, 4)] (rows 6 5)))
    (fun () -> Rune.jit lst (fun xs -> scan2 (fun c x -> let c = Nx.add c x in (c, c)) (Nx.zeros f32 [|3|]) (Nx.slice [R (1, 6); R (1, 4)] xs)) (at (rows 6 5)));
  (* two xs leaves, different dtypes and row sizes *)
  let two xs = let ints = Nx.cast Nx.int32 (Nx.mul_s xs 100.) in
    let c, ys = Rune.scan Nx.Ptree.tensor Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor
      ~f:(fun c (x, i) -> incr ran; let c = Nx.add c (Nx.add x (Nx.cast f32 (Nx.slice [R (0,3)] (Nx.concatenate ~axis:0 [i;i])))) in (c, c)) ~init:(Nx.zeros f32 [|3|]) (xs, ints) in [c; ys] in
  check "two xs leaves, int32 and f32" (fun () -> two (rows 6 3)) (fun () -> Rune.jit lst two (at (rows 6 3)));
  (* int carry *)
  let ic xs = let (c, k), ys = Rune.scan Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor Nx.Ptree.tensor
      ~f:(fun (c, k) x -> incr ran; ((Nx.add c x, Nx.add_s k 1l), Nx.cast f32 k)) ~init:(Nx.zeros f32 [|3|], Nx.scalar Nx.int32 0l) xs in [c; Nx.cast f32 k; ys] in
  check "int counter carry" (fun () -> ic (rows 6 3)) (fun () -> Rune.jit lst ic (at (rows 6 3)));
  (* f16 carry *)
  let h xs = let c, ys = Rune.scan' ~f:(fun c x -> incr ran; let c = Nx.add c (Nx.cast Nx.float16 x) in (c, Nx.cast f32 c)) ~init:(Nx.zeros Nx.float16 [|3|]) xs in [Nx.cast f32 c; ys] in
  check "f16 carry" ~tol:1e-2 (fun () -> h (rows 6 3)) (fun () -> Rune.jit lst h (at (rows 6 3)));
  (* same init for two carries *)
  let same xs = let z = Nx.ones f32 [|4|] in let (a, b), _ = Rune.scan Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun (a, b) x -> incr ran; ((Nx.add a x, Nx.mul b (Nx.add_s x 1.)), ())) ~init:(z, z) xs in [a; b] in
  check "two carries from one tensor" (fun () -> same (rows 5 4)) (fun () -> Rune.jit lst same (at (rows 5 4)));
  (* A: rng in body with traced key *)
  let drawn (k, xs) = Nx.Rng.with_key (Nx.Rng.key 0 |> fun _ -> Obj.magic k) (fun () ->
    let c, ys = Rune.scan' ~f:(fun c x -> incr ran; (Nx.add c x, Nx.add x (Nx.rand f32 [|4|]))) ~init:(Nx.zeros f32 [|4|]) xs in [c; ys]) in
  let key = (Nx.Rng.key 42 :> (int32, Nx.int32_elt) Nx.t) in
  let r = Rune.jit Nx.Ptree.(pair tensor tensor @-> returns (list tensor)) drawn (at key, at (rows 5 4)) in
  Nx.print (host (List.nth r 1));
  Nx.print (List.nth (drawn (key, rows 5 4)) 1);
  let hostmask xs = let m = Nx.less_s (Nx.arange Nx.int32 0 4 1) 2l in scan2 (fun c x -> let c = Nx.add c (Nx.where m x (Nx.zeros_like x)) in (c, Nx.mul_s c 2.)) (Nx.zeros f32 [|4|]) xs in
  check "host-computed mask in a step" (fun () -> hostmask (rows 5 4)) (fun () -> Rune.jit lst hostmask (at (rows 5 4)));
  check "draws in a staged step (traced key)" (fun () -> drawn (key, rows 5 4))
    (fun () -> Rune.jit Nx.Ptree.(pair tensor tensor @-> returns (list tensor)) drawn (at key, at (rows 5 4)));
  ()
let () =
  let key = (Nx.Rng.key 42 :> (int32, Nx.int32_elt) Nx.t) in
  let nodraw (k, xs) = Nx.Rng.with_key (Obj.magic k) (fun () -> [Nx.add xs (Nx.rand f32 [|5;4|])]) in
  check "draw without scan (traced key)" (fun () -> nodraw (key, rows 5 4))
    (fun () -> Rune.jit Nx.Ptree.(pair tensor tensor @-> returns (list tensor)) nodraw (at key, at (rows 5 4)));
  let drawn (k, xs) = Nx.Rng.with_key (Obj.magic k) (fun () ->
    let c, ys = Rune.scan' ~f:(fun c x -> incr ran; (Nx.add c x, Nx.add x (Nx.rand f32 [|4|]))) ~init:(Nx.zeros f32 [|4|]) xs in [c; ys]) in
  check "draws in a scan on the host (traced key)" (fun () -> drawn (key, rows 5 4))
    (fun () -> Rune.jit Nx.Ptree.(pair tensor tensor @-> returns (list tensor)) drawn (key, rows 5 4));
  let w = Nx.exp (Nx.sum ~axes:[0] (rows 6 4)) in
  let c = List.hd (scan2 (fun c x -> let c = Nx.add (Nx.mul c w) x in (c, Nx.mul c w)) (Nx.ones f32 [|4|]) (rows 6 4)) in
  Nx.print c

let () =
  let key = (Nx.Rng.key 42 :> (int32, Nx.int32_elt) Nx.t) in
  let drawn (k, xs) = Nx.Rng.with_key (Obj.magic k) (fun () ->
    let a = Nx.rand f32 [|4|] in
    let c, ys = Rune.scan' ~f:(fun c x -> incr ran; (Nx.add c x, Nx.add x (Nx.rand f32 [|4|]))) ~init:(Nx.zeros f32 [|4|]) xs in [a; c; ys; Nx.rand f32 [|4|]]) in
  check "draw before, in, after a scan (traced key)" (fun () -> drawn (key, rows 5 4))
    (fun () -> Rune.jit Nx.Ptree.(pair tensor tensor @-> returns (list tensor)) drawn (at key, at (rows 5 4)));
  let rot xs =
    let (a, (b, c)), _ = Rune.scan Nx.Ptree.(pair tensor (pair tensor tensor)) Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun (a, (b, c)) x -> incr ran; ((b, (Nx.add c x, a)), ())) ~init:(Nx.ones f32 [|4|], (Nx.full f32 [|4|] 2., Nx.full f32 [|4|] 3.)) xs in [a; b; c] in
  check "three-carry rotation" (fun () -> rot (rows 7 4)) (fun () -> Rune.jit lst rot (at (rows 7 4)));
  let mix xs =
    let (a, (b, c)), _ = Rune.scan Nx.Ptree.(pair tensor (pair tensor tensor)) Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun (a, (b, c)) x -> incr ran; ((Nx.add b (Nx.flip c), (Nx.add (Nx.mul_s a 0.5) x, Nx.add a b)), ())) ~init:(Nx.ones f32 [|4|], (Nx.full f32 [|4|] 2., Nx.full f32 [|4|] 3.)) xs in [a; b; c] in
  check "three carries, mixed cycles and flips" (fun () -> mix (rows 7 4)) (fun () -> Rune.jit lst mix (at (rows 7 4)));
  let many k xs =
    let init = List.init k (fun i -> Nx.full f32 [|4|] (float i *. 0.01)) in
    let cs, _ = Rune.scan Nx.Ptree.(list tensor) Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun cs x -> incr ran;
         let arr = Array.of_list cs in
         (List.mapi (fun i c -> let s = ref (Nx.add c x) in for j = 0 to i - 1 do s := Nx.add !s (Nx.mul_s arr.(j) 0.001) done; !s) cs, ())) ~init xs in cs in
  List.iter (fun k ->
    let t0 = Unix.gettimeofday () in
    check (Printf.sprintf "%d carries lower-triangular reads" k) ~tol:1e-3 (fun () -> many k (rows 3 4)) (fun () -> Rune.jit lst (many k) (at (rows 3 4)));
    Printf.printf "   took %.1fs\n%!" (Unix.gettimeofday () -. t0)) [26; 28; 30]
