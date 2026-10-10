(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Axis patterns through Nx: rearrange against a reference built from the drawn
   pattern's structure, its round trip through inverse, its views, and the
   messages of each refusal at Pattern.v and at the call. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))
let numel s = Array.fold_left ( * ) 1 s

let index s k =
  let r = Array.length s in
  let idx = Array.make r 0 and k = ref k in
  for a = r - 1 downto 0 do
    idx.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  idx

let position s idx =
  let k = ref 0 in
  Array.iteri (fun a i -> k := (!k * s.(a)) + i) idx;
  !k

let host s =
  let data = Array.init (numel s) (fun k -> Int32.of_int (k + 1)) in
  (Nx.Repr.of_array Nx.Host.v (A.of_array D.Int32 s data), data)

(* Operands laid out C-contiguous, reversed along every axis, with their axes
   reversed, or broadcast from one element. *)
type layout = Plain | Reversed | Transposed | Broadcast

let pp_layout ppf l =
  Format.pp_print_string ppf
    (match l with
    | Plain -> "plain"
    | Reversed -> "reversed"
    | Transposed -> "transposed"
    | Broadcast -> "broadcast")

let layouts =
  Gen.of_list ~pp:pp_layout [ Plain; Reversed; Transposed; Broadcast ]

(* Distinct elements of shape [s] laid out as [l] on the host, and the elements
   it reads in C order. *)
let laid l s =
  let r = Array.length s and n = numel s in
  let data = Array.init n (fun k -> Int32.of_int (k + 1)) in
  let view m a = Option.get (A.move m a) in
  let rev = Array.init r (fun i -> r - 1 - i) in
  let a =
    match l with
    | Plain -> A.of_array D.Int32 s data
    | Reversed ->
        let whole =
          Array.map
            (fun d : Nx_array.Move.range ->
              { start = max 0 (d - 1); count = d; step = -1 })
            s
        in
        view (Slice whole)
          (A.of_array D.Int32 s (Array.of_list (List.rev (Array.to_list data))))
    | Transposed ->
        let ts = Array.map (fun a -> s.(a)) rev in
        let base =
          Array.init n (fun k ->
              data.(position s (Array.map (fun a -> (index ts k).(a)) rev)))
        in
        view (Permute rev) (A.of_array D.Int32 ts base)
    | Broadcast ->
        if n = 0 then A.of_array D.Int32 s data
        else view (Broadcast s) (A.of_array D.Int32 [||] [| 1l |])
  in
  let read =
    match l with Broadcast -> Array.map (fun _ -> 1l) data | _ -> data
  in
  (Nx.Repr.of_array Nx.Host.v a, read)

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

let laws =
  group "laws"
    [
      prop "rearrange puts each name's position where the pattern says"
        (Gen.pair drawn layouts) (fun (d, l) ->
          let s = shape_of d d.left in
          cover "no element" (numel s = 0);
          cover "a split group"
            (List.exists
               (function Names (_ :: _ :: _) -> true | _ -> false)
               d.left);
          cover "a merged group"
            (List.exists
               (function Names (_ :: _ :: _) -> true | _ -> false)
               d.right);
          cover "[...]" (Array.length d.covered > 0);
          cover "a strided operand" (l = Reversed || l = Transposed);
          let x, data = laid l s in
          let y = Nx.rearrange ~sizes:d.sizes (Nx.Pattern.v (text d)) x in
          let s' = shape_of d d.right in
          equal ~msg:"shape" (array int) s' (Nx.shape y);
          equal ~msg:"elements" (array int32)
            (Array.init (numel s') (fun k -> reference d data (index s' k)))
            (elements y));
      prop "inverse undoes a rearrangement" (Gen.pair drawn layouts)
        (fun (d, l) ->
          let s = shape_of d d.left in
          let x, data = laid l s in
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
          equal ~msg:"elements" (array int32) data (elements z));
    ]

let shares x y =
  let buffer v = A.buffer (Option.get (Nx.Repr.array v)) in
  Rig.Buffer.overlaps (buffer x) (buffer y)

let views =
  group "views"
    [
      test "a split and a permutation of a contiguous value are a view"
        (fun () ->
          let x, _ = host [| 2; 3; 8 |] in
          let y =
            Nx.rearrange
              ~sizes:[ ("h", 2) ]
              (Nx.Pattern.v "b t (h d) -> b h t d")
              x
          in
          equal (array int) [| 2; 2; 3; 4 |] (Nx.shape y);
          equal bool true (shares x y));
      test "a merge strides cannot express copies" (fun () ->
          let x, _ = host [| 2; 3; 4 |] in
          let y = Nx.rearrange (Nx.Pattern.v "b t d -> (t b) d") x in
          equal bool false (shares x y);
          equal (array int) [| 6; 4 |] (Nx.shape y));
    ]

let pattern_error s msg =
  raises
    (Invalid_argument (Printf.sprintf "Nx.Pattern.v: %S: %s" s msg))
    (fun () -> Nx.Pattern.v s)

let parsing =
  group "parsing"
    [
      test "two-operand patterns of the guide parse" (fun () ->
          ignore (Nx.Pattern.v "b (kv g) i d, b kv j d -> b (kv g) i j | d");
          ignore (Nx.Pattern.v "... i k, ... k j -> ... i j | k");
          ignore (Nx.Pattern.v "i, i -> | i");
          ignore (Nx.Pattern.v "n i, n o -> i o | n"));
      cases ~name:fst "a pattern outside the grammar raises"
        [
          ("b h -> h b c", "c is not on the left");
          ("b h t d -> b t (h e)", "e is not on the left");
          ("b h t -> b h", "t is not on the right");
          ("b b -> b b", "b repeats on the left");
          ("b h -> (b h) b", "b repeats on the right");
          ("... b ... -> b", "... appears twice on the left");
          ("... b -> b", "... is on one side only");
          ("b h", "a pattern has one ->");
          ("a -> b -> c", "a pattern has one ->");
          ("b () -> b", "an empty group; write 1");
          ("b (h 1) -> b h", "a group holds names only");
          ("b (h -> b h", "a group is not closed");
          ("b h) -> b h", "a group is closed but not opened");
          ("b % -> b", "unexpected '%' at 2");
          ("2b -> 2b", "unexpected '2' at 0");
          ("b -> b | b", "| needs two operands");
          ("i j -> j i |", "| needs a name");
          ("a, b, c -> a", "a pattern has one or two operands");
          ( "b h q d, b h k d -> b h q k",
            "d is in both operands and not in the result; write it after |" );
          ("b i, b j -> b", "i is in one operand and not in the result");
          ("i k, k j -> i j | k k", "k repeats after |");
          ("i k, k j -> i j k | k", "k is summed and in the result");
          ("i k, j -> i j | k", "k is summed but not in both operands");
          ("i ..., ... j -> ... i j", "... leads all three layouts or none");
        ]
        (fun (s, msg) -> pattern_error s msg);
      test "another einsum notation raises with the pattern it means" (fun () ->
          raises
            (Invalid_argument
               "Nx.Pattern.v: \"bik,bkj->bij\" is in NumPy's einsum notation; \
                write \"b i k, b k j -> b i j | k\"") (fun () ->
              Nx.Pattern.v "bik,bkj->bij");
          raises
            (Invalid_argument
               "Nx.Pattern.v: \"ij->ji\" is in NumPy's einsum notation; write \
                \"i j -> j i\"") (fun () -> Nx.Pattern.v "ij->ji"));
      test "a one-letter-per-axis pattern that reads as words is words"
        (fun () ->
          let x, _ = host [| 2 |] in
          equal (array int) [| 2 |]
            (Nx.shape (Nx.rearrange (Nx.Pattern.v "ab->ab") x)));
      test "inverse refuses two operands" (fun () ->
          raises
            (Invalid_argument
               "Nx.Pattern.inverse: \"i k, k j -> i j | k\" has two operands")
            (fun () -> Nx.Pattern.inverse (Nx.Pattern.v "i k, k j -> i j | k")));
    ]

let call_error ?sizes s x msg =
  raises
    (Invalid_argument (Printf.sprintf "Nx.rearrange: %S%s" s msg))
    (fun () -> Nx.rearrange ?sizes (Nx.Pattern.v s) x)

let calls =
  let x, _ = host [| 2; 6; 4 |] in
  group "calls"
    [
      test "each refusal at the call names the pattern" (fun () ->
          let f32 = Nx.zeros Nx.float32 [| 2; 64; 8 |] in
          raises
            (Invalid_argument
               "Nx.rearrange: \"b h t d -> b t (h d)\": names 4 axes; the \
                operand is float32 [2; 64; 8]") (fun () ->
              Nx.rearrange (Nx.Pattern.v "b h t d -> b t (h d)") f32);
          call_error "b ... h t d -> b h t d ..." x
            ": names at least 4 axes; the operand is int32 [2; 6; 4]";
          call_error "b (h d) t -> b h d t" x
            ": (h d) has two unknown extents; give one in ~sizes";
          call_error
            ~sizes:[ ("h", 4) ]
            "b (h d) t -> b h d t" x
            ": (h d) does not divide axis 1 of extent 6";
          call_error
            ~sizes:[ ("h", 2); ("d", 2) ]
            "b (h d) t -> b h d t" x
            ": (h d) multiplies to 4, axis 1 has extent 6";
          call_error
            ~sizes:[ ("b", 3) ]
            "b h t -> t h b" x ": b = 3 in ~sizes, axis 0 has extent 2";
          call_error
            ~sizes:[ ("e", 3) ]
            "b h t -> t h b" x ": e in ~sizes is not in it";
          call_error
            ~sizes:[ ("h", 2); ("h", 2) ]
            "b (h d) t -> b h d t" x ": h is twice in ~sizes";
          call_error
            ~sizes:[ ("h", -2) ]
            "b (h d) t -> b h d t" x ": h = -2 in ~sizes is negative";
          call_error "b 1 t -> b t" x ": axis 1 has extent 6 where it has 1";
          call_error "i k, k j -> i j | k" x
            ": has two operands; use Nx.einsum or Nx.contract");
      test "units drop and add axes of extent 1" (fun () ->
          let y = Nx.rearrange (Nx.Pattern.v "b h t -> 1 b 1 (h t) 1") x in
          equal (array int) [| 1; 2; 1; 24; 1 |] (Nx.shape y);
          let z = Nx.rearrange (Nx.Pattern.v "1 b 1 n 1 -> b n") y in
          equal (array int) [| 2; 24 |] (Nx.shape z));
      test "a donated operand passes its handle through" (fun () ->
          let x =
            Nx.add (fst (host [| 2; 3 |])) (Nx.zeros Nx.int32 [| 2; 3 |])
          in
          let y = Nx.rearrange (Nx.Pattern.v "a b -> (b a)") (Nx.donate x) in
          equal (array int32)
            [| 1l; 4l; 2l; 5l; 3l; 6l |]
            (elements (Nx.copy y));
          raises_match (Exn.invalid_arg ~substring:"was donated to Nx.copy")
            (fun () -> Nx.copy x));
    ]

let () = exit (run "nx patterns" [ laws; views; parsing; calls ])
