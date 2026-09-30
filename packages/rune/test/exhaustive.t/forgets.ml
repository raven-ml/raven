(* An interpreter that forgets [Read]. *)

open Nx.Op

let run : type r. r t -> r =
 fun op ->
  match[@warning "@4@8"] op with
  | Unary _ | Binary _ | Compare _ | Where _ | Reduce _ | Scan _
  | Arg_reduce _ | Sort _ | Argsort _ | Pad _ | Cat _ | Convert _
  | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _ | Fold _
  | Matmul _ | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _
  | Lu _ | Svd _ | Eig _ | Eigh _ | Solve_triangular _ | Move _ | Place _ ->
      eval op
