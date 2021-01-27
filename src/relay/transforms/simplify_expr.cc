/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/transforms/simplify_expr.cc
 * \brief A pass for simplifying the Relay expression.
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/support/logging.h>

#include "../op/tensor/transform.h"

namespace tvm {
namespace relay {

/*!
 * \brief SimplifyReshape matches the pattern of consecutive reshape or reverse_reshape ops,
 *   and merges into one reshape op.
 */

DFPattern wildcard() { return WildcardPattern(make_object<WildcardPatternNode>()); }

DFPattern constant() { return ConstantPattern(make_object<ConstantPatternNode>()); }
class SimplifyReshape {
 public:
  SimplifyReshape() {
    x_ = WildcardPattern(make_object<WildcardPatternNode>());
    auto reshape1 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    auto reshape2 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    pattern_ = reshape1({reshape2({x_})});
  }

  Expr callback(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    auto x = node_map[x_][0];
    bool const_shape = true;
    Array<Integer> newshape;
    for (auto dim : Downcast<TensorType>(pre->checked_type())->shape) {
      if (dim.as<IntImmNode>() == nullptr) {
        const_shape = false;
        break;
      }
      newshape.push_back(Downcast<Integer>(dim));
    }
    if (const_shape) {
      return MakeReshape(x, newshape);
    }
    return post;
  }

  DFPattern pattern() const { return pattern_; }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
  /*! \brief Pattern for consecutive reshape or reverse_reshape ops */
  DFPattern pattern_;
};

class UnrollLoop {
 public:
  UnrollLoop() {
    func_var_ = wildcard();
    i_ = wildcard();
    max_count_ = wildcard();
    cond_ = wildcard();
    while_identity_ = wildcard();
    output_0_ = wildcard();
    output_1_ = wildcard();

    True = constant();

    equal_ = IsOp("equal")({cond_, True});
    less_ = IsOp("less")({i_, max_count_});
    logical_and = IsOp("logical_and")({equal_, less_});

    i_increment_ = constant();
    increment_i_ = i_ + i_increment_;
    while_increment_ = constant();
    increment_while_ = while_identity_ + while_increment_;
    cast_ = IsOp("cast")({increment_while_});
    while_limit_ = constant();
    new_cond_ = IsOp("less")({cast_, while_limit_});

    new_output_0_ = wildcard();
    new_output_1_ = wildcard();
    DFPattern tuple_0 = TuplePattern({wildcard(), new_output_0_});
    DFPattern tuple_1 = TuplePattern({wildcard(), new_output_1_});
    DFPattern concat_0 = IsOp("concatenate")({tuple_0});
    DFPattern concat_1 = IsOp("concatenate")({tuple_1});

    recursion_ = CallPattern(
        func_var_, {increment_i_, max_count_, new_cond_, increment_while_, concat_0, concat_1});
    tuple_ = TuplePattern({i_, max_count_, cond_, while_identity_, output_0_, output_1_});
    if_ = IfPattern(logical_and, recursion_, tuple_);
    func_ = FunctionPattern({i_, max_count_, cond_, while_identity_, output_0_, output_1_}, if_);
    let_ = LetPattern(func_var_, func_, func_var_);

    i_init_ = constant();
    max_count_init_ = constant();
    cond_init_ = constant();
    while_identity_init_ = constant();
    output_0_init_ = wildcard();
    output_1_init_ = wildcard();
    call_ =
        CallPattern(let_, Array<DFPattern>{i_init_, max_count_init_, cond_init_,
                                           while_identity_init_, output_0_init_, output_1_init_});
    pattern_ = call_;
  }

  DFPattern pattern() const { return pattern_; }

  class Unroller : public MixedModeMutator {
   public:
    Expr Unroll(const Expr& loop,
                std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> new_vars) {
      memo_ = new_vars;
      return VisitExpr(loop);
    }
  };

  Expr callback(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    auto get_int32 = [&node_map](const DFPattern& pattern) {
      return reinterpret_cast<int32_t*>(node_map.at(pattern)[0].as<ConstantNode>()->data->data)[0];
    };
    auto get_int64 = [&node_map](const DFPattern& pattern) {
      return reinterpret_cast<int64_t*>(node_map.at(pattern)[0].as<ConstantNode>()->data->data)[0];
    };
    auto get_float32 = [&node_map](const DFPattern& pattern) {
      return reinterpret_cast<float*>(node_map.at(pattern)[0].as<ConstantNode>()->data->data)[0];
    };
    int64_t i_init = get_int64(i_init_);
    int64_t i_increment = get_int64(i_increment_);
    int64_t i_limit = get_int64(max_count_init_);
    int64_t while_limit = get_float32(while_limit_);
    int64_t while_increment = get_int32(while_increment_);
    int64_t while_init = get_int32(while_identity_init_);
    if ((while_limit - while_init) / while_increment == 1 ||
        (i_limit - i_init) / i_increment == 1) {
      std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> new_vars;
      new_vars[node_map.at(i_)[0]] = node_map.at(i_init_)[0];
      new_vars[node_map.at(max_count_)[0]] = node_map.at(max_count_init_)[0];
      new_vars[node_map.at(cond_)[0]] = node_map.at(cond_init_)[0];
      new_vars[node_map.at(while_identity_)[0]] = node_map.at(while_identity_init_)[0];
      new_vars[node_map.at(output_0_)[0]] = node_map.at(output_0_init_)[0];
      new_vars[node_map.at(output_1_)[0]] = node_map.at(output_1_init_)[0];
      Tuple new_out({node_map.at(increment_i_)[0], node_map.at(max_count_)[0],
                     node_map.at(new_cond_)[0], node_map.at(increment_while_)[0],
                     node_map.at(new_output_0_)[0], node_map.at(new_output_1_)[0]});
      auto new_post = Unroller().Unroll(new_out, new_vars);
      return new_post;
    }
    return post;
  }

 private:
  DFPattern func_var_;
  DFPattern i_;
  DFPattern max_count_;
  DFPattern cond_;
  DFPattern while_identity_;
  DFPattern output_0_;
  DFPattern output_1_;
  DFPattern True;
  DFPattern equal_;
  DFPattern less_;
  DFPattern logical_and;
  DFPattern i_increment_;
  DFPattern increment_i_;
  DFPattern while_increment_;
  DFPattern increment_while_;
  DFPattern cast_;
  DFPattern while_limit_;
  DFPattern new_cond_;
  DFPattern new_output_0_;
  DFPattern new_output_1_;
  DFPattern recursion_;
  DFPattern tuple_;
  DFPattern if_;
  DFPattern func_;
  DFPattern let_;
  DFPattern i_init_;
  DFPattern max_count_init_;
  DFPattern cond_init_;
  DFPattern while_identity_init_;
  DFPattern output_0_init_;
  DFPattern output_1_init_;
  DFPattern call_;
  DFPattern pattern_;
};

/*!
 * \brief ExprSimplifier simplifies the Relay expression.
 */
class ExprSimplifier {
 public:
  explicit ExprSimplifier(IRModule mod) : mod_(mod) {
    auto reshape_func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = simplify_reshape_.callback(pre, post, node_map);
    };
    callbacks_.push_back(
        DFPatternCallback(simplify_reshape_.pattern(), PackedFunc(reshape_func), true));
    auto unroll_loop_func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = unroll_loop_.callback(pre, post, node_map);
    };
    callbacks_.push_back(
        DFPatternCallback(unroll_loop_.pattern(), PackedFunc(unroll_loop_func), true));
  }

  Expr Simplify(const Expr& expr) { return RewritePatterns(callbacks_, expr, mod_); }

 private:
  IRModule mod_;
  /*! \brief Simplify reshape pattern */
  SimplifyReshape simplify_reshape_;
  /*! \brief Unroll Loop pattern */
  UnrollLoop unroll_loop_;
  /*! \brief Callbacks for expr simplification */
  Array<DFPatternCallback> callbacks_;
};

Expr SimplifyExpr(const Expr& expr, const IRModule& mod) {
  auto out = ExprSimplifier(mod).Simplify(expr);
  return out;
}

namespace transform {

Pass SimplifyExpr() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SimplifyExpr(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyExpr").set_body_typed(SimplifyExpr);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
