type ('a, _) t = { w : 'a } constraint 'a = int [@@deriving ptree]
