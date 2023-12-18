  $ cat code.toy
  def f1() {
    var a = 1 + 2;
    var b = (1 + 1);
    var c = 2 + 2 * 2;
    var d = (2 + 2) * 2;
    var e = ((2 + 1) * (2 + (3 + 2))) * (2 + 1);
    return e;
  }
  
  def f2() {
    var a = [1] + [2];
    var b = ([1] + [1]);
    var c = [2] + [2] * [2];
    var d = ([2] + [2]) * [2];
    var e = (([2] + [1]) * ([2] + ([3] + [2]))) * ([2] + [1]);
    return e;
  }
  $ dune exec -- toy -emit ast -f code.toy
  [(Ast.Function ((Ast.Prototype ("f1", [])),
      [(Ast.VarDecl ("a", [||], (Ast.BinOp (`Add, (Ast.Num 1.), (Ast.Num 2.)))
          ));
        (Ast.VarDecl ("b", [||], (Ast.BinOp (`Add, (Ast.Num 1.), (Ast.Num 1.)))
           ));
        (Ast.VarDecl ("c", [||],
           (Ast.BinOp (`Add, (Ast.Num 2.),
              (Ast.BinOp (`Mul, (Ast.Num 2.), (Ast.Num 2.)))))
           ));
        (Ast.VarDecl ("d", [||],
           (Ast.BinOp (`Mul, (Ast.BinOp (`Add, (Ast.Num 2.), (Ast.Num 2.))),
              (Ast.Num 2.)))
           ));
        (Ast.VarDecl ("e", [||],
           (Ast.BinOp (`Mul,
              (Ast.BinOp (`Mul, (Ast.BinOp (`Add, (Ast.Num 2.), (Ast.Num 1.))),
                 (Ast.BinOp (`Add, (Ast.Num 2.),
                    (Ast.BinOp (`Add, (Ast.Num 3.), (Ast.Num 2.)))))
                 )),
              (Ast.BinOp (`Add, (Ast.Num 2.), (Ast.Num 1.)))))
           ));
        (Ast.Return (Some (Ast.Var "e")))]
      ));
    (Ast.Function ((Ast.Prototype ("f2", [])),
       [(Ast.VarDecl ("a", [||],
           (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 1.)])),
              (Ast.Literal ([|1|], [(Ast.Num 2.)]))))
           ));
         (Ast.VarDecl ("b", [||],
            (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 1.)])),
               (Ast.Literal ([|1|], [(Ast.Num 1.)]))))
            ));
         (Ast.VarDecl ("c", [||],
            (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 2.)])),
               (Ast.BinOp (`Mul, (Ast.Literal ([|1|], [(Ast.Num 2.)])),
                  (Ast.Literal ([|1|], [(Ast.Num 2.)]))))
               ))
            ));
         (Ast.VarDecl ("d", [||],
            (Ast.BinOp (`Mul,
               (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 2.)])),
                  (Ast.Literal ([|1|], [(Ast.Num 2.)])))),
               (Ast.Literal ([|1|], [(Ast.Num 2.)]))))
            ));
         (Ast.VarDecl ("e", [||],
            (Ast.BinOp (`Mul,
               (Ast.BinOp (`Mul,
                  (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 2.)])),
                     (Ast.Literal ([|1|], [(Ast.Num 1.)])))),
                  (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 2.)])),
                     (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 3.)])),
                        (Ast.Literal ([|1|], [(Ast.Num 2.)]))))
                     ))
                  )),
               (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 2.)])),
                  (Ast.Literal ([|1|], [(Ast.Num 1.)]))))
               ))
            ));
         (Ast.Return (Some (Ast.Var "e")))]
       ))
    ]
