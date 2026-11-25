(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.mli's Axis patterns: rearrange against a reference built from the drawn
   pattern's structure, its round trip through inverse, the extents every arity
   resolves, its views, and the raises of Pattern.v, inverse and rearrange,
   named as far as nx.mli states. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype

let elements = Nx.to_array
let numel s = Array.fold_left ( * ) 1 s

let position s idx =
  let k = ref 0 in
  Array.iteri (fun a i -> k := (!k * s.(a)) + i) idx;
  !k

(* Distinct elements of shape [s] laid out as [l] on the host, and the elements
   it reads in C order. *)
let laid_ints l s =
  laid l D.Int32 s (Array.init (numel s) (fun k -> Int32.of_int (k + 1)))

let host s = fst (laid_ints C_order s)

(* A drawn one-operand pattern: named axes [0 .. n - 1] of [extents], and each
   side's items: a group of names (one name written alone), a unit, or the axes
   [...] covers, of [covered] extents. *)
type item = Names of int list | Unit | Covered

type drawn = {
  extents : int array;
  covered : int array;
  left : item list;
  right : item list;
  sizes : (string * int) list;
}

let names = [| "b"; "h"; "t"; "d"; "c"; "kv_2" |]

let item_text = function
  | Names [ i ] -> names.(i)
  | Names is -> "(" ^ String.concat " " (List.map (fun i -> names.(i)) is) ^ ")"
  | Unit -> "1"
  | Covered -> "..."

let side l = String.concat " " (List.map item_text l)
let text d = side d.left ^ " -> " ^ side d.right

(* The extents an item spans. *)
let span d = function
  | Names is -> [ List.fold_left (fun a i -> a * d.extents.(i)) 1 is ]
  | Unit -> [ 1 ]
  | Covered -> Array.to_list d.covered

let shape_of d l = Array.of_list (List.concat_map (span d) l)

(* [atoms] cut into runs of one to three, each a group, with units and [...]
   placed among them. *)
let items atoms ~units ~covered =
  let open Gen in
  let rec runs = function
    | [] -> constant []
    | atoms ->
        let* k = int_range 1 (min 3 (List.length atoms)) in
        let+ rest = runs (List.filteri (fun i _ -> i >= k) atoms) in
        Names (List.filteri (fun i _ -> i < k) atoms) :: rest
  in
  let* groups = runs atoms in
  let extra =
    List.init units (fun _ -> Unit) @ if covered then [ Covered ] else []
  in
  let* order =
    permutation (List.init (List.length groups + List.length extra) Fun.id)
  in
  let all = Array.of_list (groups @ extra) in
  (* Groups keep their order; units and [...] go where [order] puts them. *)
  let slots = List.map (fun i -> i < List.length groups) order in
  let g = ref 0 and e = ref (List.length groups) in
  constant
    (List.map
       (fun is_group ->
         if is_group then (
           incr g;
           all.(!g - 1))
         else (
           incr e;
           all.(!e - 1)))
       slots)

let drawn =
  let open Gen in
  with_pp
    (fun ppf d ->
      Format.fprintf ppf "%S ~sizes:[%s]" (text d)
        (String.concat "; "
           (List.map (fun (n, e) -> Printf.sprintf "%s=%d" n e) d.sizes)))
    (let* n = int_range 0 5 in
     let* extents = array ~size:(constant n) (int_range 0 3) in
     let* covered = bool in
     let* k = int_range 0 2 in
     let* cov =
       array ~size:(constant (if covered then k else 0)) (int_range 1 3)
     in
     let* units_l = int_range 0 1 in
     let* units_r = int_range 0 1 in
     let* left = items (List.init n Fun.id) ~units:units_l ~covered in
     let* perm = permutation (List.init n Fun.id) in
     let* right = items perm ~units:units_r ~covered in
     let+ unknown = array ~size:(constant (List.length left)) (int_range 0 2) in
     (* Each group of two or more names gives every extent in ~sizes but at most
        one, which the quotient gives where the others' product is not zero. *)
     let sizes =
       List.concat
         (List.mapi
            (fun j -> function
              | Names (_ :: _ :: _ as is) ->
                  let left_out = List.nth is (unknown.(j) mod List.length is) in
                  let others = List.filter (fun i -> i <> left_out) is in
                  let product =
                    List.fold_left (fun a i -> a * extents.(i)) 1 others
                  in
                  let given = if product = 0 then is else others in
                  List.map (fun i -> (names.(i), extents.(i))) given
              | Names _ | Unit | Covered -> [])
            left)
     in
     { extents; covered = cov; left; right; sizes })

(* The element the result of [d] holds at [i]: each name's position read off the
   result's index, then the operand's index composed from them. *)
let reference d data i =
  let at = Array.make (Array.length d.extents) 0 in
  let cov = Array.make (Array.length d.covered) 0 in
  let k = ref 0 in
  List.iter
    (function
      | Names is ->
          let sub =
            index (Array.of_list (List.map (fun i -> d.extents.(i)) is)) i.(!k)
          in
          List.iteri (fun j a -> at.(a) <- sub.(j)) is;
          incr k
      | Unit -> incr k
      | Covered ->
          Array.iteri (fun j _ -> cov.(j) <- i.(!k + j)) cov;
          k := !k + Array.length cov)
    d.right;
  let operand =
    List.concat_map
      (function
        | Names is ->
            [
              position
                (Array.of_list (List.map (fun i -> d.extents.(i)) is))
                (Array.of_list (List.map (fun a -> at.(a)) is));
            ]
        | Unit -> [ 0 ]
        | Covered -> Array.to_list cov)
      d.left
  in
  data.(position (shape_of d d.left) (Array.of_list operand))

let law_rearrange (d, l) =
  let s = shape_of d d.left in
  cover "no element" (numel s = 0);
  cover "a split group"
    (List.exists (function Names (_ :: _ :: _) -> true | _ -> false) d.left);
  cover "a merged group"
    (List.exists (function Names (_ :: _ :: _) -> true | _ -> false) d.right);
  cover "[...]" (Array.length d.covered > 0);
  cover "a strided operand"
    (l = Reversed || l = Axes_reversed || l = Every_other);
  let x, data = laid_ints l s in
  let y = Nx.rearrange ~sizes:d.sizes (Nx.Pattern.v (text d)) x in
  let s' = shape_of d d.right in
  equal ~msg:"shape" (array int) s' (Nx.shape y);
  equal ~msg:"elements" (array int32)
    (Array.init (numel s') (fun k -> reference d data (index s' k)))
    (elements y)

let law_inverse (d, l) =
  let s = shape_of d d.left in
  let x, data = laid_ints l s in
  let p = Nx.Pattern.v (text d) in
  let y = Nx.rearrange ~sizes:d.sizes p x in
  (* The inverse splits the groups [p] merged: their extents. *)
  let merged =
    List.concat_map
      (function
        | Names (_ :: _ :: _ as is) ->
            List.map (fun i -> (names.(i), d.extents.(i))) is
        | Names _ | Unit | Covered -> [])
      d.right
  in
  let z = Nx.rearrange ~sizes:merged (Nx.Pattern.inverse p) y in
  equal ~msg:"shape" (array int) s (Nx.shape z);
  equal ~msg:"elements" (array int32) data (elements z)

(* A drawn pattern of one to four operands that sums nothing: the first operand
   holds every name, each other some of them, [...] leads every layout or none.
   [sizes] leaves out of each group of two or more names at most one, which its
   quotient gives, and every name an operand holds alone, which that axis
   gives. *)
type joint = {
  extents : int array;
  covered : int array;
  operands : item list list;
  result : item list;
  sizes : (string * int) list;
}

let joint_text j =
  String.concat ", " (List.map side j.operands) ^ " -> " ^ side j.result

let alone j i = List.exists (List.mem (Names [ i ])) j.operands

let joint =
  let open Gen in
  with_pp
    (fun ppf j ->
      Format.fprintf ppf "%S ~sizes:[%s] ~covered:[%s]" (joint_text j)
        (String.concat "; "
           (List.map (fun (n, e) -> Printf.sprintf "%s=%d" n e) j.sizes))
        (String.concat "; " (Array.to_list (Array.map string_of_int j.covered))))
    (let* n = int_range 1 5 in
     let* extents = array ~size:(constant n) (int_range 0 3) in
     let* covered = bool in
     let* k = int_range 1 2 in
     let* cov =
       array ~size:(constant (if covered then k else 0)) (int_range 1 3)
     in
     let layout atoms =
       let* units = int_range 0 1 in
       let+ l = items atoms ~units ~covered:false in
       if covered then Covered :: l else l
     in
     let* arity = int_range 1 4 in
     let* first = layout (List.init n Fun.id) in
     let* others =
       list
         ~size:(constant (arity - 1))
         (let* perm = permutation (List.init n Fun.id) in
          let* held = int_range 1 n in
          layout (List.filteri (fun i _ -> i < held) perm))
     in
     let* perm = permutation (List.init n Fun.id) in
     let* result = layout perm in
     let operands = first :: others in
     let groups =
       List.concat_map
         (List.filter_map (function
           | Names (_ :: _ :: _ as is) -> Some is
           | Names _ | Unit | Covered -> None))
         operands
     in
     let+ picks = list ~size:(constant (List.length groups)) (int_range 0 2) in
     let j = { extents; covered = cov; operands; result; sizes = [] } in
     let given =
       List.concat
         (List.map2
            (fun is pick ->
              let left_out = List.nth is (pick mod List.length is) in
              let others = List.filter (fun i -> i <> left_out) is in
              let product =
                List.fold_left (fun a i -> a * extents.(i)) 1 others
              in
              if product = 0 then is else others)
            groups picks)
     in
     let given =
       List.filter (fun i -> not (alone j i)) (List.sort_uniq Int.compare given)
     in
     { j with sizes = List.map (fun i -> (names.(i), extents.(i))) given })

(* Whether a group of two or more names in [j] holds a name [sizes] leaves out
   that another operand holds alone, if [by_axis], or that none does. *)
let solved_by j by_axis =
  List.exists
    (List.exists (function
      | Names (_ :: _ :: _ as is) ->
          List.exists
            (fun i ->
              (not (List.mem_assoc names.(i) j.sizes)) && alone j i = by_axis)
            is
      | Names _ | Unit | Covered -> false))
    j.operands

let law_resolve j =
  let arity = List.length j.operands in
  cover "one operand" (arity = 1);
  cover "two operands" (arity = 2);
  cover "three operands or more" (arity >= 3);
  cover "a group's quotient" (solved_by j false);
  cover "another operand's axis" (solved_by j true);
  cover "[...]" (Array.length j.covered > 0);
  cover "no element" (Array.exists (( = ) 0) j.extents);
  let shape l =
    Array.of_list
      (List.concat_map
         (function
           | Names is -> [ List.fold_left (fun a i -> a * j.extents.(i)) 1 is ]
           | Unit -> [ 1 ]
           | Covered -> Array.to_list j.covered)
         l)
  in
  let p = Nx.Pattern.v (joint_text j) and sizes = j.sizes in
  let xs = List.map (fun l -> Nx.zeros Nx.float32 (shape l)) j.operands in
  let want = shape j.result in
  match xs with
  | [ x ] ->
      equal ~msg:"rearrange" (array int) want
        (Nx.shape (Nx.rearrange ~sizes p x));
      equal ~msg:"einsum" (array int) want (Nx.shape (Nx.einsum ~sizes p xs))
  | [ a; b ] ->
      equal ~msg:"contract" (array int) want
        (Nx.shape (Nx.contract ~sizes Nx.float32 p a b));
      equal ~msg:"einsum" (array int) want (Nx.shape (Nx.einsum ~sizes p xs))
  | xs ->
      equal ~msg:"einsum" (array int) want (Nx.shape (Nx.einsum ~sizes p xs))

(* One operand whose group [(b h)] leaves [h] to the quotient of its axis. *)
let quotient =
  {
    extents = [| 2; 3 |];
    covered = [||];
    operands = [ [ Names [ 0; 1 ] ] ];
    result = [ Names [ 0 ]; Names [ 1 ] ];
    sizes = [ ("b", 2) ];
  }

(* The message [f] raises. *)
let said f =
  match ignore (f ()) with
  | () -> "raised nothing"
  | exception Invalid_argument m -> m

(* A raise of [by] naming the pattern [s], as nx.mli states. *)
let names_pattern ~by s f =
  match ignore (f ()) with
  | () -> failf "%s raised nothing" by
  | exception Invalid_argument m ->
      starts_with ~prefix:(Printf.sprintf "%s: %S" by s) m

let outside_grammar =
  [
    "b h -> h b c";
    "b h t d -> b t (h e)";
    "b h t -> b h";
    "b b -> b b";
    "b h -> (b h) b";
    "... b ... -> b";
    "... b -> b";
    "b h";
    "a -> b -> c";
    "b () -> b";
    "b (h 1) -> b h";
    "b (h -> b h";
    "b h) -> b h";
    "b % -> b";
    "2b -> 2b";
    "b -> b | b";
    "i j -> j i |";
    "b h q d, b h k d -> b h q k";
    "i k, k j, k -> i j";
    "a, b, c -> a";
    "b i i -> b";
    "b i, b j -> b";
    "i k, k j -> i j | k k";
    "i k, k j -> i j k | k";
    "i, j -> i j | k";
    "i i -> | j";
    "i ..., ... j -> ... i j";
    "... i, ... j, k -> ... i j k";
  ]

let pattern_v =
  group "Pattern.v"
    [
      cases "parses patterns of one or more operands, sums and repeats"
        ~id:(Printf.sprintf "%S")
        [
          "b (kv g) i d, b kv j d -> b (kv g) i j | d";
          "... i k, ... k j -> ... i j | k";
          "i, i -> | i";
          "n i, n o -> i o | n";
          "i j, j k, k -> i | j k";
          "b i k, b k, b k j -> b i j | k";
          "... i, ... i, ... i, ... i -> ... | i";
          "z a a -> z a";
          "i j -> i | j";
          "i k, j -> i j | k";
          "(a b) a, b -> a b";
        ] (fun s -> ignore (Nx.Pattern.v s));
      cases "raises naming a pattern outside the grammar"
        ~id:(Printf.sprintf "%S") outside_grammar (fun s ->
          names_pattern ~by:"Nx.Pattern.v" s (fun () -> Nx.Pattern.v s));
      test "raises with these reasons" (fun () ->
          expect
            (String.concat "\n"
               (List.map
                  (fun s -> said (fun () -> Nx.Pattern.v s))
                  outside_grammar))
          @@ __POS_OF__
               {|
            Nx.Pattern.v: "b h -> h b c": c is not on the left
            Nx.Pattern.v: "b h t d -> b t (h e)": e is not on the left
            Nx.Pattern.v: "b h t -> b h": t is not on the right
            Nx.Pattern.v: "b b -> b b": b repeats in the result
            Nx.Pattern.v: "b h -> (b h) b": b repeats on the right
            Nx.Pattern.v: "... b ... -> b": ... appears twice on the left
            Nx.Pattern.v: "... b -> b": ... is on one side only
            Nx.Pattern.v: "b h": a pattern has one ->
            Nx.Pattern.v: "a -> b -> c": a pattern has one ->
            Nx.Pattern.v: "b () -> b": an empty group; write 1
            Nx.Pattern.v: "b (h 1) -> b h": a group holds names only
            Nx.Pattern.v: "b (h -> b h": a group is not closed
            Nx.Pattern.v: "b h) -> b h": a group is closed but not opened
            Nx.Pattern.v: "b % -> b": unexpected '%' at 2
            Nx.Pattern.v: "2b -> 2b": unexpected '2' at 0
            Nx.Pattern.v: "b -> b | b": b is summed and in the result
            Nx.Pattern.v: "i j -> j i |": | needs a name
            Nx.Pattern.v: "b h q d, b h k d -> b h q k": d is in 2 operands and not in the result; write it after |
            Nx.Pattern.v: "i k, k j, k -> i j": k is in 3 operands and not in the result; write it after |
            Nx.Pattern.v: "a, b, c -> a": b is in one operand and not in the result; write it after |
            Nx.Pattern.v: "b i i -> b": i is in one operand and not in the result; write it after |
            Nx.Pattern.v: "b i, b j -> b": i is in one operand and not in the result; write it after |
            Nx.Pattern.v: "i k, k j -> i j | k k": k repeats after |
            Nx.Pattern.v: "i k, k j -> i j k | k": k is summed and in the result
            Nx.Pattern.v: "i, j -> i j | k": k is summed but in no operand
            Nx.Pattern.v: "i i -> | j": j is summed but in no operand
            Nx.Pattern.v: "i ..., ... j -> ... i j": ... leads every layout or none
            Nx.Pattern.v: "... i, ... j, k -> ... i j k": ... leads every layout or none
            |});
      cases "raises on NumPy's einsum notation with the pattern it means"
        ~id:fst
        [
          ("bik,bkj->bij", "b i k, b k j -> b i j | k");
          ("ij->ji", "i j -> j i");
          ("ij,jk,k->i", "i j, j k, k -> i | j k");
          ("zaa->za", "z a a -> z a");
          ("ij->i", "i j -> i | j");
        ]
        (fun (s, meant) ->
          let m = said (fun () -> Nx.Pattern.v s) in
          starts_with ~prefix:(Printf.sprintf "Nx.Pattern.v: %S" s) m;
          contains ~sub:(Printf.sprintf "write %S" meant) m);
      test "reads a one-letter-per-axis pattern that reads as words as words"
        (fun () ->
          equal (array int) [| 2 |]
            (Nx.shape (Nx.rearrange (Nx.Pattern.v "ab->ab") (host [| 2 |]))));
      prop "resolves its names' extents alike at every arity"
        ~examples:[ quotient ] joint law_resolve;
    ]

let pattern_inverse =
  group "Pattern.inverse"
    [
      prop "undoes a rearrangement" (Gen.pair drawn layouts) law_inverse;
      cases "raises naming the pattern" ~id:(Printf.sprintf "%S")
        [ "i i -> i"; "i j -> i | j"; "i k, k j -> i j | k" ] (fun s ->
          let p = Nx.Pattern.v s in
          names_pattern ~by:"Nx.Pattern.inverse" s (fun () ->
              Nx.Pattern.inverse p));
    ]

(* Whether [y]'s memory is [x]'s. *)
let held v = Option.get (Nx.Repr.array ~by:"test" v)
let shares x y = Rig.Buffer.overlaps (A.buffer (held x)) (A.buffer (held y))

let laid_out =
  Testable.make
    ~pp:(fun ppf v -> A.Layout.pp ppf (A.layout (held v)))
    ~equal:( == )

(* The calls that raise, each a pattern, its sizes and the operand [int32 [2; 6;
   4]]. *)
let refused =
  [
    ("b ... h t d -> b h t d ...", []);
    ("b (h d) t -> b h d t", []);
    ("b (h d) t -> b h d t", [ ("h", 4) ]);
    ("b (h d) t -> b h d t", [ ("h", 2); ("d", 2) ]);
    ("b h t -> t h b", [ ("b", 3) ]);
    ("b h t -> t h b", [ ("e", 3) ]);
    ("b (h d) t -> b h d t", [ ("h", 2); ("h", 2) ]);
    ("b (h d) t -> b h d t", [ ("h", -2) ]);
    ("b (h d) t -> b h d t", [ ("h", max_int); ("d", 2) ]);
    ("b 1 t -> b t", []);
    ("i k, k j -> i j | k", []);
    ("b h h -> b h", []);
    ("b h t -> b h | t", []);
    ("i k, k j, j -> i | k j", []);
  ]

let pp_refused (s, sizes) =
  Printf.sprintf "%S ~sizes:[%s]" s
    (String.concat "; "
       (List.map (fun (n, e) -> Printf.sprintf "%s=%d" n e) sizes))

let call (s, sizes) () =
  Nx.rearrange ~sizes (Nx.Pattern.v s) (host [| 2; 6; 4 |])

let rearrange =
  group "rearrange"
    [
      prop "puts each name's position where the pattern says"
        (Gen.pair drawn layouts) law_rearrange;
      test "drops and adds axes of extent 1 at its units" (fun () ->
          let x = host [| 2; 6; 4 |] in
          let y = Nx.rearrange (Nx.Pattern.v "b h t -> 1 b 1 (h t) 1") x in
          equal (array int) [| 1; 2; 1; 24; 1 |] (Nx.shape y);
          let z = Nx.rearrange (Nx.Pattern.v "1 b 1 n 1 -> b n") y in
          equal (array int) [| 2; 24 |] (Nx.shape z));
      test "makes a view of a split and a permutation of a contiguous value"
        (fun () ->
          let x = host [| 2; 3; 8 |] in
          let y =
            Nx.rearrange
              ~sizes:[ ("h", 2) ]
              (Nx.Pattern.v "b t (h d) -> b h t d")
              x
          in
          equal (array int) [| 2; 2; 3; 4 |] (Nx.shape y);
          satisfies ~claim:"a view over the operand's memory" laid_out
            (shares x) y);
      test "copies a merge strides cannot express" (fun () ->
          let x = host [| 2; 3; 4 |] in
          let y = Nx.rearrange (Nx.Pattern.v "b t d -> (t b) d") x in
          equal (array int) [| 6; 4 |] (Nx.shape y);
          satisfies ~claim:"memory of its own" laid_out
            (fun y -> not (shares x y))
            y);
      (* A split and a permutation of a view allocated 621 words a call before
         its extents were resolved in one place, 535 after. *)
      test "allocates at most 372 words for a split and a permutation of a view"
        (fun () ->
          let x = host [| 1; 64; 64 |] in
          let p = Nx.Pattern.v "b t (h d) -> b h t d" in
          let sizes = [ ("h", 16) ] in
          at_most float_exact ~than:372.
            (words (fun () -> Nx.rearrange ~sizes p x)));
      test "passes a donated operand's handle through" (fun () ->
          let x = Nx.add (host [| 2; 3 |]) (Nx.zeros Nx.int32 [| 2; 3 |]) in
          let y = Nx.rearrange (Nx.Pattern.v "a b -> (b a)") (Nx.donate x) in
          equal (array int32)
            [| 1l; 4l; 2l; 5l; 3l; 6l |]
            (elements (Nx.copy y));
          raises_match (Exn.invalid_arg ~sub:"was donated to Nx.copy")
            (fun () -> Nx.copy y));
      cases "raises naming the pattern" ~id:pp_refused refused
        (fun (s, sizes) -> names_pattern ~by:"Nx.rearrange" s (call (s, sizes)));
      test "raises naming the pattern and the operand's rank" (fun () ->
          let x = Nx.zeros Nx.float32 [| 2; 64; 8 |] in
          names_pattern ~by:"Nx.rearrange" "b h t d -> b t (h d)" (fun () ->
              Nx.rearrange (Nx.Pattern.v "b h t d -> b t (h d)") x));
      test "raises with these reasons" (fun () ->
          expect
            (String.concat "\n" (List.map (fun c -> said (call c)) refused))
          @@ __POS_OF__
               {|
            Nx.rearrange: "b ... h t d -> b h t d ..." names at least 4 axes for x; x is int32 [2; 6; 4]
            Nx.rearrange: "b (h d) t -> b h d t": (h d) in x leaves h and d unknown; give one in ~sizes
            Nx.rearrange: "b (h d) t -> b h d t": (h d) is 6 in x, which 4 does not divide
            Nx.rearrange: "b (h d) t -> b h d t": (h d) is 6 in x, and its names multiply to 4
            Nx.rearrange: "b h t -> t h b": b is 3 in ~sizes and 2 in x
            Nx.rearrange: "b h t -> t h b": ~sizes names e, which it lacks
            Nx.rearrange: "b (h d) t -> b h d t": h is twice in ~sizes
            Nx.rearrange: "b (h d) t -> b h d t": h = -2 in ~sizes is negative
            Nx.rearrange: "b (h d) t -> b h d t": (h d) is 6 in x, and its names multiply to more than an int counts
            Nx.rearrange: "b 1 t -> b t": axis 1 of x is 1 in the pattern and 6 in x
            Nx.rearrange: "i k, k j -> i j | k": has 2 operands; use Nx.einsum
            Nx.rearrange: "b h h -> b h": takes a diagonal or sums within one operand; use Nx.einsum
            Nx.rearrange: "b h t -> b h | t": takes a diagonal or sums within one operand; use Nx.einsum
            Nx.rearrange: "i k, k j, j -> i | k j": has 3 operands; use Nx.einsum
            |});
    ]

let () = exit (run "nx patterns" [ pattern_v; pattern_inverse; rearrange ])
