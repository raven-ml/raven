type t = unit

let create () = ()

let tokenize () text =
  if String.length text = 0 then []
  else
    let chars = ref [] in
    let offset = ref 0 in
    String.iter
      (fun c ->
        let char_str = String.make 1 c in
        let id = Char.code c in
        chars := (id, char_str, (!offset, !offset + 1)) :: !chars;
        incr offset)
      text;
    List.rev !chars

let token_to_id () token =
  if String.length token = 1 then Some (Char.code token.[0]) else None

let id_to_token () id =
  if id >= 0 && id <= 255 then Some (String.make 1 (Char.chr id)) else None

let get_vocab () = []
let get_vocab_size () = 256 (* All ASCII characters *)
let save () ~folder:_ () = []
