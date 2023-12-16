  $ cat code.toy
  # User defined generic function that operates on unknown shaped arguments.
  def multiply_transpose(a, b) {
    return transpose(a) * transpose(b);
  }
  
  def main() {
    # Define a variable `a` with shape <2, 3>, initialized with the literal value.
    # The shape is inferred from the supplied literal.
    var a = [[1, 2, 3], [4, 5, 6]];
    # b is identical to a, the literal array is implicitly reshaped: defining new
    # variables is the way to reshape arrays (element count in literal must match
    # the size of specified shape).
    var b<2, 3> = [1, 2, 3, 4, 5, 6];
  
    # This call will specialize `multiply_transpose` with <2, 3> for both
    # arguments and deduce a return type of <2, 2> in initialization of `c`.
    var c = multiply_transpose(a, b);
    # A second call to `multiply_transpose` with <2, 3> for both arguments will
    # reuse the previously specialized and inferred version and return `<2, 2>`
    var d = multiply_transpose(b, a);
    # A new call with `<2, 2>` for both dimension will trigger another
    # specialization of `multiply_transpose`.
    var e = multiply_transpose(b, c);
    # Finally, calling into `multiply_transpose` with incompatible shape will
    # trigger a shape inference error.
    var f = multiply_transpose(transpose(a), c);
  }
  $ dune exec -- toy -emit ast -f code.toy
  [(Ast.Function ((Ast.Prototype ("multiply_transpose", ["a"; "b"])),
      [(Ast.Return
          (Some (Ast.BinOp ('*', (Ast.Call ("transpose", [(Ast.Var "a")])),
                   (Ast.Call ("transpose", [(Ast.Var "b")]))))))
        ]
      ));
    (Ast.Function ((Ast.Prototype ("main", [])),
       [(Ast.VarDecl ("a", [||],
           (Ast.Literal ([|2; 3|],
              [(Ast.Literal ([|3|], [(Ast.Num 1.); (Ast.Num 2.); (Ast.Num 3.)]
                  ));
                (Ast.Literal ([|3|], [(Ast.Num 4.); (Ast.Num 5.); (Ast.Num 6.)]
                   ))
                ]
              ))
           ));
         (Ast.VarDecl ("b", [|2; 3|],
            (Ast.Literal ([|6|],
               [(Ast.Num 1.); (Ast.Num 2.); (Ast.Num 3.); (Ast.Num 4.);
                 (Ast.Num 5.); (Ast.Num 6.)]
               ))
            ));
         (Ast.VarDecl ("c", [||],
            (Ast.Call ("multiply_transpose", [(Ast.Var "a"); (Ast.Var "b")]))));
         (Ast.VarDecl ("d", [||],
            (Ast.Call ("multiply_transpose", [(Ast.Var "b"); (Ast.Var "a")]))));
         (Ast.VarDecl ("e", [||],
            (Ast.Call ("multiply_transpose", [(Ast.Var "b"); (Ast.Var "c")]))));
         (Ast.VarDecl ("f", [||],
            (Ast.Call ("multiply_transpose",
               [(Ast.Call ("transpose", [(Ast.Var "a")])); (Ast.Var "c")]))
            ))
         ]
       ))
    ]
