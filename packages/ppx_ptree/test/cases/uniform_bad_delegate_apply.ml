type 'a sub = { x : 'a }
and 'a t = { v : 'a option sub; w : 'a } [@@deriving ptree]
