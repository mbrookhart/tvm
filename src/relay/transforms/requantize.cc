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
  software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file requantize.cc
 * \brief Push quantization ops through the graph
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../ir/indexed_graph.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {
namespace quantize {

bool is_op(const Expr& node, const Expr& op) {
  if (auto call_node = node.as<CallNode>()) {
    if (call_node->op == op) {
      return true;
    }
  }
  return false;
}

class Requantizer {
 public:
  Expr Requantize(const Expr& expr) {
    graph_ = CreateIndexedGraph(expr);
    // traverse the graph in reverse topological order
    for (size_t i = graph_.topological_order_.size(); i-- != 0;) {
      auto& node = graph_.topological_order_[i];
      if (is_op(node->ref_, dequantize_op)) {
        std::vector<Expr> terminating_quantizes;
        bool removeable = true;
        for (auto output : node->outputs_) {
          if (removeable) {
            Expr termination = FindQuantizeUser(output);
            if (termination.defined()) {
              terminating_quantizes.push_back(termination);
            } else {
              removeable = false;
            }
          }
        }
        if (removeable) {
          for (size_t i = 0; i < node->outputs_.size(); ++i) {
            quantize_pairs_[node->outputs_[i]->ref_] = {0, terminating_quantizes[i]};
            removable_quantizes.insert(terminating_quantizes[i]);
          }
        }
      }
    }

    return RequantizeMutator(this).Mutate(expr);
  }

 protected:
  Expr FindQuantizeUser(IndexedGraph<Expr>::Node* node) {
    if (is_op(node->ref_, quantize_op)) {
      return node->ref_;
    } else if (node->ref_.as<CallNode>() == nullptr ||
               !fquantize_rewrite_.count(Downcast<Op>(node->ref_.as<CallNode>()->op)) ||
               node->outputs_.size() == 0) {
      return Expr();
    }
    Expr out = FindQuantizeUser(node->outputs_[0]);
    for (size_t i = 1; i < node->outputs_.size(); ++i) {
      Expr tmp = FindQuantizeUser(node->outputs_[i]);
      if (tmp != out) {
        return Expr();
      }
    }
    return out;
  }
  OpAttrMap<FTVMQuantizeRewrite> fquantize_rewrite_ =
      Op::GetAttrMap<FTVMQuantizeRewrite>("FTVMQuantizeRewrite");
  std::unordered_map<Expr, std::pair<size_t, Expr>, ObjectPtrHash, ObjectPtrEqual> quantize_pairs_;
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> removable_quantizes;

  IndexedGraph<Expr> graph_;
  Expr dequantize_op = Op::Get("qnn.dequantize");
  Expr quantize_op = Op::Get("qnn.quantize");

  class RequantizeMutator : public MixedModeMutator {
   public:
    RequantizeMutator(Requantizer* parent) : parent_(parent) {}

    Expr Rewrite_(const CallNode* call_node, const Expr& post) {
      const CallNode* post_call = post.as<CallNode>();
      Op op = Downcast<Op>(post_call->op);
      if (parent_->quantize_pairs_.count(GetRef<Expr>(call_node))) {
        auto pair = parent_->quantize_pairs_[GetRef<Expr>(call_node)];
        // Record the current quantize for use when processing greylisted ops
        parent_quantize_ = pair.second;

        const CallNode* quantize = pair.second.as<CallNode>();
        auto attrs = quantize->attrs.as<qnn::QuantizeAttrs>();
        Array<Expr> new_args;
        for (size_t i = 0; i < post_call->args.size(); ++i) {
          if (i == pair.first) {
            new_args.push_back(qnn::MakeQuantize(post_call->args[i], quantize->args[1],
                                                 quantize->args[2], attrs->axis, attrs->out_dtype));
          } else {
            new_args.push_back(post_call->args[i]);
          }
        }
        Expr new_post = Call(call_node->op, new_args, call_node->attrs);
        if (parent_quantize_.defined() && parent_->fquantize_rewrite_.count(op)) {
          return parent_->fquantize_rewrite_[op](new_post, parent_quantize_.as<CallNode>()->args[1],
                                                 parent_quantize_.as<CallNode>()->args[2]);
        }
        return new_post;
      } else if (parent_->removable_quantizes.count(GetRef<Expr>(call_node))) {
        return post_call->args[0];
      } else if (parent_quantize_.defined() && parent_->fquantize_rewrite_.count(op)) {
        return parent_->fquantize_rewrite_[op](post, parent_quantize_.as<CallNode>()->args[1],
                                               parent_quantize_.as<CallNode>()->args[2]);
      } else if (call_node->op == Op::Get("qnn.dequantize")) {
        parent_quantize_ = Expr();
      }
      return post;
    }

   protected:
    Requantizer* parent_;
    Expr parent_quantize_;
  };
};

TVM_REGISTER_GLOBAL("relay.transform.quantize.requantize").set_body_typed([](const Expr& expr) {
  return Requantizer().Requantize(expr);
});

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
