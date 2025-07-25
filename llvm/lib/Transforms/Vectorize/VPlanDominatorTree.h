//===-- VPlanDominatorTree.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements dominator tree analysis for a single level of a VPlan's
/// H-CFG.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANDOMINATORTREE_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANDOMINATORTREE_H

#include "VPlan.h"
#include "VPlanCFG.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

namespace llvm {

template <> struct DomTreeNodeTraits<VPBlockBase> {
  using NodeType = VPBlockBase;
  using NodePtr = VPBlockBase *;
  using ParentPtr = VPlan *;

  static NodePtr getEntryNode(ParentPtr Parent) { return Parent->getEntry(); }
  static ParentPtr getParent(NodePtr B) { return B->getPlan(); }
};

/// Template specialization of the standard LLVM dominator tree utility for
/// VPBlockBases.
class VPDominatorTree : public DominatorTreeBase<VPBlockBase, false> {
  using Base = DominatorTreeBase<VPBlockBase, false>;

public:
  VPDominatorTree() = default;
  explicit VPDominatorTree(VPlan &Plan) { recalculate(Plan); }

  /// Returns true if \p A properly dominates \p B.
  using Base::properlyDominates;
  bool properlyDominates(const VPRecipeBase *A, const VPRecipeBase *B);
};

using VPDomTreeNode = DomTreeNodeBase<VPBlockBase>;

/// Template specializations of GraphTraits for VPDomTreeNode.
template <>
struct GraphTraits<VPDomTreeNode *>
    : public DomTreeGraphTraitsBase<VPDomTreeNode,
                                    VPDomTreeNode::const_iterator> {};

template <>
struct GraphTraits<const VPDomTreeNode *>
    : public DomTreeGraphTraitsBase<const VPDomTreeNode,
                                    VPDomTreeNode::const_iterator> {};
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANDOMINATORTREE_H
