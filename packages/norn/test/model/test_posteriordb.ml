(* Law 10, Recovery's, on two posteriors: each sampler on dogs, NUTS on
   eight_schools_noncentered, whose scale and effects reach the reference
   through M.constrain and a bijector's Jacobian, and a NUTS run too short for
   its diagnostics. Each cell is sized so that it recovers, or for the short run
   is flagged, on every key from 1 to 40, so that a change to the random stream
   or to rounding does not fail a correct sampler. One Bonferroni bound holds
   the false-alarm rate over the six recovering cells' 50 comparisons at 1%: z =
   3.72, and t = 4.88 for SMC and nested sampling's 16 replicates.
   posteriordb_sweep holds the law over every posterior.

   Nested sampling at 25 live points is not held to the shrinkage of ln X. One
   that shrinks it by 1 / (n - j + 2) at the j-th deleted point, one past the
   true 1 / (n - j + 1), fails this test on key 1 and on key 35, and passes on
   the other 38 of keys 1 to 40: its largest error has a median of 2.7 standard
   errors over the keys, where the correct sampler's has 1.8 and reaches 3.7,
   against the bound of 4.88. *)

module P = Posteriordb
module R = Recovery

let posteriors = P.all "../golden/posteriordb.golden"
let post name = List.find (fun (P.Posterior p) -> p.name = name) posteriors
let dogs = post "dogs-dogs"
let eight_schools = post "eight_schools-eight_schools_noncentered"

let () =
  exit
    (Windtrap.run "Norn posteriordb"
       (R.recovers
          [
            (* R-hat up to 1.020 on keys 1 to 40 at 100 draws. *)
            R.chain_cell (R.nuts ~draws:300 ()) dogs;
            R.chain_cell (R.hmc ~draws:50) dogs;
            (* R-hat 1.011 on one of keys 1 to 40 at 500 draws. *)
            R.chain_cell (R.ensemble ~ensembles:2 ~draws:800) dogs;
            R.replicated_cell (R.smc ~replicates:16 ~draws:100) dogs;
            R.replicated_cell (R.nested ~replicates:16 ~draws:25) dogs;
            (* At the default acceptance a transition or two diverges on 17 of
               keys 1 to 40, and at 0.9 on 3. *)
            R.chain_cell (R.nuts ~accept:0.95 ~draws:300 ()) eight_schools;
          ]
       @ [
           (* 160 draws: every element's effective size is below the 400 the
              summary asks, on each of keys 1 to 40. *)
           R.flagged (R.chain_cell (R.nuts ~draws:20 ()) dogs);
         ]))
