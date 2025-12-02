(* Stateless, splittable RNG keys shared across Nx and Rune *)

type key = int

let key seed = Stdlib.abs seed land 0x7FFFFFFF (* Ensure positive 31-bit int *)

(* MurmurHash-inspired integer hash for better distribution *)
let hash_int x =
  let open Int32 in
  let x = of_int x in
  let x = logxor x (shift_right_logical x 16) in
  let x = mul x 0x85ebca6bl in
  let x = logxor x (shift_right_logical x 13) in
  let x = mul x 0xc2b2ae35l in
  let x = logxor x (shift_right_logical x 16) in
  to_int (logand x 0x7FFFFFFFl)

let split ?(n = 2) k = Array.init n (fun i -> hash_int ((k * (n + 1)) + i + 1))
let fold_in k data = hash_int (k lxor data)
let to_int k = k

module Generator = struct
  type t = { mutable key : key }

  let create ?key:init () =
    let key = match init with Some k -> k | None -> key (Random.bits ()) in
    { key }

  let next t =
    match split ~n:2 t.key with
    | [| next_key; subkey |] ->
        t.key <- next_key;
        subkey
    | _ -> assert false

  let current_key t = t.key
end
