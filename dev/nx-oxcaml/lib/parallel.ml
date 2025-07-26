(* Parallel execution utilities for the OxCaml backend *)

(* Thread pool management *)
type pool = {
  n_threads : int;
  (* For now, we'll use a simple placeholder. In a real implementation,
     we'd integrate with Domain or other parallelism libraries *)
}

let global_pool = ref None

let get_or_setup_pool ?(n_threads = 8) () =
  match !global_pool with
  | Some pool -> pool
  | None ->
      let pool = { n_threads } in
      global_pool := Some pool;
      pool

let parallel_for _pool ~start ~stop ~body =
  (* For now, sequential execution. Later we can use Domain.spawn for real parallelism *)
  for i = start to stop - 1 do
    body i
  done

let parallel_chunks _pool ~n_elements ~chunk_size ~body =
  let n_chunks = (n_elements + chunk_size - 1) / chunk_size in
  for chunk_idx = 0 to n_chunks - 1 do
    let start = chunk_idx * chunk_size in
    let stop = min (start + chunk_size) n_elements in
    body ~chunk_idx ~start ~stop
  done