//===- GCF.cpp - Structured Control Flow Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GCF/IR/GCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::gcf;

#include "mlir/Dialect/GCF/IR/GCFOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GCF/IR/GCFOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// GCFDialect
//===----------------------------------------------------------------------===//

void GCFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/GCF/IR/GCFOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/GCF/IR/GCFOpsAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

MutableOperandRange BreakOp::getLabelMutableSuccessorOperands(CFGLabel label) {
  return MutableOperandRange(*this);
}

void BreakOp::getSuccessorLabels(std::optional<ArrayRef<Attribute>> operands,
                                 SmallVectorImpl<CFGLabel> &labels) {
  labels.push_back(BreakLabel::get(getContext()));
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

MutableOperandRange
ContinueOp::getLabelMutableSuccessorOperands(CFGLabel label) {
  return MutableOperandRange(*this);
}

void ContinueOp::getSuccessorLabels(std::optional<ArrayRef<Attribute>> operands,
                                    SmallVectorImpl<CFGLabel> &labels) {
  labels.push_back(ContinueLabel::get(getContext()));
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

LogicalResult ForOp::verify() {
  // Check that the number of init args and op results is the same.
  if (getInitArgs().size() != getNumResults())
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");

  return success();
}

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer' values
/// are regular SSA values.
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

void ForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();

  printInitializationList(p, getRegionIterArgs(), getInitArgs(), " iter_args");
  if (!getInitArgs().empty())
    p << " -> (" << getInitArgs().getTypes() << ')';
  p << ' ';
  if (Type t = getInductionVar().getType(); !t.isIndex())
    p << " : " << t << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!getInitArgs().empty());
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  Type type;

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  // Parse the induction variable followed by '='.
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);

  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("iter_args"));
  if (hasIterArgs) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
  }

  if (regionArgs.size() != result.types.size() + 1)
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  // Parse optional type, else assume Index.
  if (parser.parseOptionalColon())
    type = builder.getIndexType();
  else if (parser.parseType(type))
    return failure();

  // Set block argument types, so that they are known when parsing the region.
  regionArgs.front().type = type;
  for (auto [iterArg, type] :
       llvm::zip_equal(llvm::drop_begin(regionArgs), result.types))
    iterArg.type = type;

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // Resolve input operands. This should be done after parsing the region to
  // catch invalid IR where operands were defined inside of the region.
  if (parser.resolveOperand(lb, type, result.operands) ||
      parser.resolveOperand(ub, type, result.operands) ||
      parser.resolveOperand(step, type, result.operands))
    return failure();
  if (hasIterArgs) {
    for (auto argOperandType : llvm::zip_equal(llvm::drop_begin(regionArgs),
                                               operands, result.types)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                result.operands))
        return failure();
    }
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

Block::BlockArgListType ForOp::getRegionIterArgs() {
  return getBody()->getArguments().drop_front(getNumInductionVars());
}

std::optional<APInt> ForOp::getConstantStep() {
  IntegerAttr step;
  if (matchPattern(getStep(), m_Constant(&step)))
    return step.getValue();
  return {};
}

Speculation::Speculatability ForOp::getSpeculatability() {
  // `scf.for (I = Start; I < End; I += 1)` terminates for all values of Start
  // and End.
  if (auto constantStep = getConstantStep())
    if (*constantStep == 1)
      return Speculation::RecursivelySpeculatable;

  // For Step != 1, the loop may not terminate.  We can add more smarts here if
  // needed.
  return Speculation::NotSpeculatable;
}

OperandRange ForOp::getOnEntrySuccessorOperands(CFGBranchPoint point) {
  return getInitArgs();
}

void ForOp::getOnEntrySuccessors(std::optional<ArrayRef<Attribute>> operands,
                                 SmallVectorImpl<CFGSuccessor> &successors) {
  successors.push_back(CFGSuccessor(getBodyRegion()));
  successors.push_back(CFGSuccessor(CFGBranchPoint::parent()));
}

void ForOp::getLabelSuccessors(CFGLabel label,
                               SmallVectorImpl<CFGSuccessor> &successors) {
  if (isa<YieldToParentAttr, ContinueLabel>(label)) {
    successors.push_back(CFGSuccessor(getBodyRegion()));
    successors.push_back(CFGSuccessor(CFGBranchPoint::parent()));
    return;
  }
  if (!isa<BreakLabel>(label))
    return;
  successors.push_back(CFGSuccessor(CFGBranchPoint::parent()));
}

void ForOp::getAcceptedTerminators(
    const DenseSet<std::pair<CFGLabel, CFGTerminatorOpInterface>> &terminators,
    SmallVectorImpl<std::pair<CFGLabel, CFGTerminatorOpInterface>> &accepted) {
  for (auto [label, term] : terminators) {
    if (!isa<YieldToParentAttr, ContinueLabel, BreakLabel>(label))
      continue;
    accepted.push_back({label, term});
  }
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult
IfOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                       IfOp::Adaptor adaptor,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  if (adaptor.getRegions().empty())
    return failure();
  Region *r = &adaptor.getThenRegion();
  if (r->empty())
    return failure();
  Block &b = r->front();
  if (b.empty())
    return failure();
  auto yieldOp = llvm::dyn_cast<YieldOp>(b.back());
  if (!yieldOp)
    return success();
  TypeRange types = yieldOp.getOperandTypes();
  llvm::append_range(inferredReturnTypes, types);
  return success();
}

LogicalResult IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");
  return success();
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << " " << getCondition();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

Block *IfOp::thenBlock() { return &getThenRegion().back(); }
YieldOp IfOp::thenYield() { return cast<YieldOp>(&thenBlock()->back()); }
Block *IfOp::elseBlock() {
  Region &r = getElseRegion();
  if (r.empty())
    return nullptr;
  return &r.back();
}
YieldOp IfOp::elseYield() { return cast<YieldOp>(&elseBlock()->back()); }

OperandRange IfOp::getOnEntrySuccessorOperands(CFGBranchPoint point) {
  return getOperation()->getOperands();
}

void IfOp::getOnEntrySuccessors(std::optional<ArrayRef<Attribute>> operands,
                                SmallVectorImpl<CFGSuccessor> &successors) {
  successors.push_back(CFGSuccessor(getThenRegion()));
  if (!getElseRegion().empty()) {
    successors.push_back(CFGSuccessor(getElseRegion()));
    return;
  }
  successors.push_back(CFGSuccessor(CFGBranchPoint::parent()));
}

void IfOp::getLabelSuccessors(CFGLabel label,
                              SmallVectorImpl<CFGSuccessor> &successors) {
  if (isa<YieldToParentAttr>(label)) {
    successors.push_back(CFGSuccessor(CFGBranchPoint::parent()));
    return;
  }
}

void IfOp::getAcceptedTerminators(
    const DenseSet<std::pair<CFGLabel, CFGTerminatorOpInterface>> &terminators,
    SmallVectorImpl<std::pair<CFGLabel, CFGTerminatorOpInterface>> &accepted) {
  for (auto [label, term] : terminators) {
    if (!isa<YieldToParentAttr>(label))
      continue;
    accepted.push_back({label, term});
  }
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

MutableOperandRange YieldOp::getLabelMutableSuccessorOperands(CFGLabel label) {
  return MutableOperandRange(*this);
}

void YieldOp::getSuccessorLabels(std::optional<ArrayRef<Attribute>> operands,
                                 SmallVectorImpl<CFGLabel> &labels) {
  labels.push_back(YieldToParentAttr::get(getContext()));
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/GCF/IR/GCFOps.cpp.inc"
