//===- ControlFlowInterfaces.cpp - ControlFlow Interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ControlFlowInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.cpp.inc"

SuccessorOperands::SuccessorOperands(MutableOperandRange forwardedOperands)
    : producedOperandCount(0), forwardedOperands(std::move(forwardedOperands)) {
}

SuccessorOperands::SuccessorOperands(unsigned int producedOperandCount,
                                     MutableOperandRange forwardedOperands)
    : producedOperandCount(producedOperandCount),
      forwardedOperands(std::move(forwardedOperands)) {}

//===----------------------------------------------------------------------===//
// BranchOpInterface
//===----------------------------------------------------------------------===//

/// Returns the `BlockArgument` corresponding to operand `operandIndex` in some
/// successor if 'operandIndex' is within the range of 'operands', or
/// std::nullopt if `operandIndex` isn't a successor operand index.
std::optional<BlockArgument>
detail::getBranchSuccessorArgument(const SuccessorOperands &operands,
                                   unsigned operandIndex, Block *successor) {
  OperandRange forwardedOperands = operands.getForwardedOperands();
  // Check that the operands are valid.
  if (forwardedOperands.empty())
    return std::nullopt;

  // Check to ensure that this operand is within the range.
  unsigned operandsStart = forwardedOperands.getBeginOperandIndex();
  if (operandIndex < operandsStart ||
      operandIndex >= (operandsStart + forwardedOperands.size()))
    return std::nullopt;

  // Index the successor.
  unsigned argIndex =
      operands.getProducedOperandCount() + operandIndex - operandsStart;
  return successor->getArgument(argIndex);
}

/// Verify that the given operands match those of the given successor block.
LogicalResult
detail::verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                      const SuccessorOperands &operands) {
  // Check the count.
  unsigned operandCount = operands.size();
  Block *destBB = op->getSuccessor(succNo);
  if (operandCount != destBB->getNumArguments())
    return op->emitError() << "branch has " << operandCount
                           << " operands for successor #" << succNo
                           << ", but target block has "
                           << destBB->getNumArguments();

  // Check the types.
  for (unsigned i = operands.getProducedOperandCount(); i != operandCount;
       ++i) {
    if (!cast<BranchOpInterface>(op).areTypesCompatible(
            operands[i].getType(), destBB->getArgument(i).getType()))
      return op->emitError() << "type mismatch for bb argument #" << i
                             << " of successor #" << succNo;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface
//===----------------------------------------------------------------------===//

static InFlightDiagnostic &printRegionEdgeName(InFlightDiagnostic &diag,
                                               RegionBranchPoint sourceNo,
                                               RegionBranchPoint succRegionNo) {
  diag << "from ";
  if (Region *region = sourceNo.getRegionOrNull())
    diag << "Region #" << region->getRegionNumber();
  else
    diag << "parent operands";

  diag << " to ";
  if (Region *region = succRegionNo.getRegionOrNull())
    diag << "Region #" << region->getRegionNumber();
  else
    diag << "parent results";
  return diag;
}

/// Verify that types match along all region control flow edges originating from
/// `sourcePoint`. `getInputsTypesForRegion` is a function that returns the
/// types of the inputs that flow to a successor region.
static LogicalResult
verifyTypesAlongAllEdges(Operation *op, RegionBranchPoint sourcePoint,
                         function_ref<FailureOr<TypeRange>(RegionBranchPoint)>
                             getInputsTypesForRegion) {
  auto regionInterface = cast<RegionBranchOpInterface>(op);

  SmallVector<RegionSuccessor, 2> successors;
  regionInterface.getSuccessorRegions(sourcePoint, successors);

  for (RegionSuccessor &succ : successors) {
    FailureOr<TypeRange> sourceTypes = getInputsTypesForRegion(succ);
    if (failed(sourceTypes))
      return failure();

    TypeRange succInputsTypes = succ.getSuccessorInputs().getTypes();
    if (sourceTypes->size() != succInputsTypes.size()) {
      InFlightDiagnostic diag = op->emitOpError(" region control flow edge ");
      return printRegionEdgeName(diag, sourcePoint, succ)
             << ": source has " << sourceTypes->size()
             << " operands, but target successor needs "
             << succInputsTypes.size();
    }

    for (const auto &typesIdx :
         llvm::enumerate(llvm::zip(*sourceTypes, succInputsTypes))) {
      Type sourceType = std::get<0>(typesIdx.value());
      Type inputType = std::get<1>(typesIdx.value());
      if (!regionInterface.areTypesCompatible(sourceType, inputType)) {
        InFlightDiagnostic diag = op->emitOpError(" along control flow edge ");
        return printRegionEdgeName(diag, sourcePoint, succ)
               << ": source type #" << typesIdx.index() << " " << sourceType
               << " should match input type #" << typesIdx.index() << " "
               << inputType;
      }
    }
  }
  return success();
}

/// Verify that types match along control flow edges described the given op.
LogicalResult detail::verifyTypesAlongControlFlowEdges(Operation *op) {
  auto regionInterface = cast<RegionBranchOpInterface>(op);

  auto inputTypesFromParent = [&](RegionBranchPoint point) -> TypeRange {
    return regionInterface.getEntrySuccessorOperands(point).getTypes();
  };

  // Verify types along control flow edges originating from the parent.
  if (failed(verifyTypesAlongAllEdges(op, RegionBranchPoint::parent(),
                                      inputTypesFromParent)))
    return failure();

  auto areTypesCompatible = [&](TypeRange lhs, TypeRange rhs) {
    if (lhs.size() != rhs.size())
      return false;
    for (auto types : llvm::zip(lhs, rhs)) {
      if (!regionInterface.areTypesCompatible(std::get<0>(types),
                                              std::get<1>(types))) {
        return false;
      }
    }
    return true;
  };

  // Verify types along control flow edges originating from each region.
  for (Region &region : op->getRegions()) {

    // Since there can be multiple terminators implementing the
    // `RegionBranchTerminatorOpInterface`, all should have the same operand
    // types when passing them to the same region.

    SmallVector<RegionBranchTerminatorOpInterface> regionReturnOps;
    for (Block &block : region)
      if (!block.empty())
        if (auto terminator =
                dyn_cast<RegionBranchTerminatorOpInterface>(block.back()))
          regionReturnOps.push_back(terminator);

    // If there is no return-like terminator, the op itself should verify
    // type consistency.
    if (regionReturnOps.empty())
      continue;

    auto inputTypesForRegion =
        [&](RegionBranchPoint point) -> FailureOr<TypeRange> {
      std::optional<OperandRange> regionReturnOperands;
      for (RegionBranchTerminatorOpInterface regionReturnOp : regionReturnOps) {
        auto terminatorOperands = regionReturnOp.getSuccessorOperands(point);

        if (!regionReturnOperands) {
          regionReturnOperands = terminatorOperands;
          continue;
        }

        // Found more than one ReturnLike terminator. Make sure the operand
        // types match with the first one.
        if (!areTypesCompatible(regionReturnOperands->getTypes(),
                                terminatorOperands.getTypes())) {
          InFlightDiagnostic diag = op->emitOpError(" along control flow edge");
          return printRegionEdgeName(diag, region, point)
                 << " operands mismatch between return-like terminators";
        }
      }

      // All successors get the same set of operand types.
      return TypeRange(regionReturnOperands->getTypes());
    };

    if (failed(verifyTypesAlongAllEdges(op, region, inputTypesForRegion)))
      return failure();
  }

  return success();
}

/// Stop condition for `traverseRegionGraph`. The traversal is interrupted if
/// this function returns "true" for a successor region. The first parameter is
/// the successor region. The second parameter indicates all already visited
/// regions.
using StopConditionFn = function_ref<bool(Region *, ArrayRef<bool> visited)>;

/// Traverse the region graph starting at `begin`. The traversal is interrupted
/// if `stopCondition` evaluates to "true" for a successor region. In that case,
/// this function returns "true". Otherwise, if the traversal was not
/// interrupted, this function returns "false".
static bool traverseRegionGraph(Region *begin,
                                StopConditionFn stopConditionFn) {
  auto op = cast<RegionBranchOpInterface>(begin->getParentOp());
  SmallVector<bool> visited(op->getNumRegions(), false);
  visited[begin->getRegionNumber()] = true;

  // Retrieve all successors of the region and enqueue them in the worklist.
  SmallVector<Region *> worklist;
  auto enqueueAllSuccessors = [&](Region *region) {
    SmallVector<RegionSuccessor> successors;
    op.getSuccessorRegions(region, successors);
    for (RegionSuccessor successor : successors)
      if (!successor.isParent())
        worklist.push_back(successor.getSuccessor());
  };
  enqueueAllSuccessors(begin);

  // Process all regions in the worklist via DFS.
  while (!worklist.empty()) {
    Region *nextRegion = worklist.pop_back_val();
    if (stopConditionFn(nextRegion, visited))
      return true;
    if (visited[nextRegion->getRegionNumber()])
      continue;
    visited[nextRegion->getRegionNumber()] = true;
    enqueueAllSuccessors(nextRegion);
  }

  return false;
}

/// Return `true` if region `r` is reachable from region `begin` according to
/// the RegionBranchOpInterface (by taking a branch).
static bool isRegionReachable(Region *begin, Region *r) {
  assert(begin->getParentOp() == r->getParentOp() &&
         "expected that both regions belong to the same op");
  return traverseRegionGraph(begin,
                             [&](Region *nextRegion, ArrayRef<bool> visited) {
                               // Interrupt traversal if `r` was reached.
                               return nextRegion == r;
                             });
}

/// Return `true` if `a` and `b` are in mutually exclusive regions.
///
/// 1. Find the first common of `a` and `b` (ancestor) that implements
///    RegionBranchOpInterface.
/// 2. Determine the regions `regionA` and `regionB` in which `a` and `b` are
///    contained.
/// 3. Check if `regionA` and `regionB` are mutually exclusive. They are
///    mutually exclusive if they are not reachable from each other as per
///    RegionBranchOpInterface::getSuccessorRegions.
bool mlir::insideMutuallyExclusiveRegions(Operation *a, Operation *b) {
  assert(a && "expected non-empty operation");
  assert(b && "expected non-empty operation");

  auto branchOp = a->getParentOfType<RegionBranchOpInterface>();
  while (branchOp) {
    // Check if b is inside branchOp. (We already know that a is.)
    if (!branchOp->isProperAncestor(b)) {
      // Check next enclosing RegionBranchOpInterface.
      branchOp = branchOp->getParentOfType<RegionBranchOpInterface>();
      continue;
    }

    // b is contained in branchOp. Retrieve the regions in which `a` and `b`
    // are contained.
    Region *regionA = nullptr, *regionB = nullptr;
    for (Region &r : branchOp->getRegions()) {
      if (r.findAncestorOpInRegion(*a)) {
        assert(!regionA && "already found a region for a");
        regionA = &r;
      }
      if (r.findAncestorOpInRegion(*b)) {
        assert(!regionB && "already found a region for b");
        regionB = &r;
      }
    }
    assert(regionA && regionB && "could not find region of op");

    // `a` and `b` are in mutually exclusive regions if both regions are
    // distinct and neither region is reachable from the other region.
    return regionA != regionB && !isRegionReachable(regionA, regionB) &&
           !isRegionReachable(regionB, regionA);
  }

  // Could not find a common RegionBranchOpInterface among a's and b's
  // ancestors.
  return false;
}

bool RegionBranchOpInterface::isRepetitiveRegion(unsigned index) {
  Region *region = &getOperation()->getRegion(index);
  return isRegionReachable(region, region);
}

bool RegionBranchOpInterface::hasLoop() {
  SmallVector<RegionSuccessor> entryRegions;
  getSuccessorRegions(RegionBranchPoint::parent(), entryRegions);
  for (RegionSuccessor successor : entryRegions)
    if (!successor.isParent() &&
        traverseRegionGraph(successor.getSuccessor(),
                            [](Region *nextRegion, ArrayRef<bool> visited) {
                              // Interrupt traversal if the region was already
                              // visited.
                              return visited[nextRegion->getRegionNumber()];
                            }))
      return true;
  return false;
}

Region *mlir::getEnclosingRepetitiveRegion(Operation *op) {
  while (Region *region = op->getParentRegion()) {
    op = region->getParentOp();
    if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op))
      if (branchOp.isRepetitiveRegion(region->getRegionNumber()))
        return region;
  }
  return nullptr;
}

Region *mlir::getEnclosingRepetitiveRegion(Value value) {
  Region *region = value.getParentRegion();
  while (region) {
    Operation *op = region->getParentOp();
    if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op))
      if (branchOp.isRepetitiveRegion(region->getRegionNumber()))
        return region;
    region = op->getParentRegion();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CFGOpInterface
//===----------------------------------------------------------------------===//

namespace {
struct CFGBuilder {
  CFGBuilder(CFGContext &context) : context(context) {}
  CFGPoint *build(Operation *op);

private:
  CFGOp *build(RegionBranchOpInterface op);
  CFGTerminator *build(RegionBranchTerminatorOpInterface op);
  CFGOp *build(GeneralRegionBranchOpInterface op);
  CFGTerminator *build(GeneralRegionBranchTerminatorOpInterface op);
  CFGContext &context;
  llvm::ScopedHashTable<CFGLabel, CFGOp *> table;
};
} // namespace

CFGPoint *CFGBuilder::build(Operation *op) {
  CFGPoint *point{};
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op))
    point = build(branch);
  else if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op))
    point = build(terminator);
  else if (auto branch = dyn_cast<GeneralRegionBranchOpInterface>(op))
    point = build(branch);
  else if (auto terminator =
               dyn_cast<GeneralRegionBranchTerminatorOpInterface>(op))
    point = build(terminator);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &op : block)
        (void)build(&op);
  return point;
}

CFGOp *CFGBuilder::build(RegionBranchOpInterface op) {
  auto point = new CFGOp(op);
  context.insert(point);
  SmallVector<CFGRegion *> regions;
  for (Region &region : op->getRegions()) {
    auto point = new CFGRegion(&region);
    context.insert(point);
    regions.push_back(point);
  }
  SmallVector<RegionSuccessor> parentSuccessors;
  op.getSuccessorRegions(RegionBranchPoint::parent(), parentSuccessors);
  for (RegionSuccessor successor : parentSuccessors) {
    if (successor.isParent()) {
      point->pushSuccessor(point);
      continue;
    }
    point->pushSuccessor(regions[successor.getSuccessor()->getRegionNumber()]);
  }
  return point;
}

CFGTerminator *CFGBuilder::build(RegionBranchTerminatorOpInterface op) {
  auto point = new CFGTerminator(op);
  context.insert(point);
  if (!isa<RegionBranchOpInterface>(op->getParentOp()))
    return nullptr;
  SmallVector<Attribute> attrs(op->getOperands().size());
  SmallVector<RegionSuccessor> successors;
  op.getSuccessorRegions(attrs, successors);
  for (RegionSuccessor successor : successors) {
    CFGPoint *succPoint{};
    if (successor.isParent())
      succPoint = context.lookup(op->getParentOp());
    else
      succPoint = context.lookup(successor.getSuccessor());
    point->pushSuccessor(cast<CFGFlowPoint>(succPoint));
  }
  return point;
}

CFGOp *CFGBuilder::build(GeneralRegionBranchOpInterface op) { return nullptr; }

CFGTerminator *CFGBuilder::build(GeneralRegionBranchTerminatorOpInterface op) {
  return nullptr;
}

CFGPoint *mlir::buildOpCFG(Operation *op, CFGContext &context) {
  return CFGBuilder(context).build(op);
}

//===----------------------------------------------------------------------===//
// CFGPrinter
//===----------------------------------------------------------------------===//

namespace {
struct CFGPrinter {
  CFGPrinter(llvm::raw_ostream &stream, CFGContext &context)
      : stream(stream), context(context) {}

  void dumpGraph(Operation *op);

private:
  /// Dump the graph header.
  void dumpHeader();

  /// Dump an op.
  void dumpOp(Operation *op);

  /// Dump a block.
  void dumpBlock(Block *block);

  void dumpSuccessors(Operation *op, Block *parent);

  //===--------------------------------------------------------------------===//
  // Utility functions
  //===--------------------------------------------------------------------===//

  /// Print an indented line.
  llvm::raw_ostream &print() {
    stream.printIndent();
    return stream.getOStream();
  }

  /// Return unique labels.
  uint32_t getId(Operation *op);
  uint32_t getId(Region *region, Block *block);
  std::string getLabel(Operation *op);
  std::string getLabel(Block *block);
  std::string getHeader(Operation *op);
  std::string getHeader(Block *block);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  llvm::ScopedPrinter stream;
  llvm::DenseMap<Operation *, uint32_t> opLabels;
  llvm::DenseMap<std::pair<Region *, Block *>, uint32_t> blockLabels;
  llvm::SmallPtrSet<void *, 8> visited;
  CFGContext &context;
};
} // namespace

uint32_t CFGPrinter::getId(Operation *op) {
  uint32_t &label = opLabels[op];
  if (label == 0)
    label = opLabels.size();
  return label;
}

uint32_t CFGPrinter::getId(Region *region, Block *block) {
  uint32_t &label = blockLabels[{region, block}];
  if (label == 0)
    label = blockLabels.size();
  return label;
}

std::string CFGPrinter::getLabel(Operation *op) {
  return llvm::formatv("o{0}", getId(op)).str();
}

std::string CFGPrinter::getLabel(Block *block) {
  Region *region = block->getParent();
  Operation *op = region->getParentOp();
  return llvm::formatv("b{0}_{1}_{2}", getId(op), region->getRegionNumber(),
                       getId(region, block))
      .str();
}

void CFGPrinter::dumpGraph(Operation *op) {
  dumpHeader();
  dumpOp(op);
  stream.unindent();
  print() << "}\n";
}

void CFGPrinter::dumpHeader() {
  print() << "digraph {\n";
  stream.indent();
  print() << "rankdir=LR;\n";
  print() << "node [ shape=record ];\n";
}

std::string CFGPrinter::getHeader(Operation *op) {
  return llvm::formatv("Op[{0}]: {1}", getId(op), op->getName().getStringRef())
      .str();
}
std::string CFGPrinter::getHeader(Block *block) {
  Region *region = block->getParent();
  Operation *op = region->getParentOp();
  return llvm::formatv("Block[{0}, {1}, {2}]", getId(op),
                       region->getRegionNumber(), getId(region, block))
      .str();
}

void CFGPrinter::dumpOp(Operation *op) {
  if (!visited.insert(op).second)
    return;
  std::string label = getLabel(op);
  print() << label << "[\n";
  stream.indent();
  print() << "label = \"<root>" << getHeader(op) << "\"\n";
  stream.indent();
  for (Region &region : op->getRegions()) {
    if (region.empty() || region.front().empty())
      continue;
    print() << "+ \"| <" << getLabel(&region.front()) << "> "
            << getHeader(&region.front()) << "\"\n";
  }
  stream.unindent(2);
  print() << "];\n";
  for (Region &region : op->getRegions()) {
    if (region.empty() || region.front().empty())
      continue;
    Block *entry = &region.front();
    std::string blockLabel = getLabel(entry);
    print() << label << ":" << blockLabel << " -> " << blockLabel << ":root;\n";
    dumpBlock(entry);
  }
  dumpSuccessors(op, nullptr);
}

void CFGPrinter::dumpBlock(Block *block) {
  if (!visited.insert(block).second)
    return;
  std::string label = getLabel(block);
  print() << label << "[\n";
  stream.indent();
  print() << "label = \"<root>" << getHeader(block) << "\"\n";
  stream.indent();
  for (Operation &op : *block) {
    print() << "+ \"| <" << getLabel(&op) << "> " << getHeader(&op) << "\"\n";
  }
  stream.unindent(2);
  print() << "];\n";
  for (Operation &op : *block) {
    if (op.getNumRegions() == 0 ||
        llvm::all_of(op.getRegions(),
                     [](Region &region) { return region.empty(); }))
      continue;
    std::string opLabel = getLabel(&op);
    print() << label << ":" << opLabel << " -> " << opLabel << ":root;\n";
    dumpOp(&op);
  }
  Operation &maybeTerminator = block->back();
  if (!maybeTerminator.hasTrait<OpTrait::IsTerminator>() ||
      (maybeTerminator.getNumRegions() != 0 &&
       llvm::any_of(maybeTerminator.getRegions(),
                    [](Region &region) { return !region.empty(); })))
    return;
  dumpSuccessors(&maybeTerminator, block);
}

void CFGPrinter::dumpSuccessors(Operation *op, Block *parent) {
  CFGPoint *point = context.lookup(op);
  std::string label = getLabel(op);
  if (parent)
    label = getLabel(parent) + ":" + label;
  else
    label = label + ":root";
  auto printSuccessors = [&](CFGSuccessorRange range) {
    for (CFGFlowPoint *point : range) {
      if (auto op = dyn_cast<CFGOp>(point)) {
        print() << label << ":root -> " << getLabel(op->getOp())
                << ":root [color=\"green\"];\n";
      } else if (auto region = dyn_cast<CFGRegion>(point)) {
        print() << label << ":root -> "
                << getLabel(&region->getRegion()->front())
                << ":root [color=\"blue\"];\n";
      }
    }
  };
  if (CFGOp *cfgOp = dyn_cast_or_null<CFGOp>(point))
    printSuccessors(cfgOp->getSuccessors());
  else if (CFGTerminator *term = dyn_cast_or_null<CFGTerminator>(point))
    printSuccessors(term->getSuccessors());
  for (Block *successor : op->getSuccessors()) {
    print() << label << " -> " << getLabel(successor)
            << ":root [color=\"red\"];\n";
    dumpBlock(successor);
  }
}

void mlir::printOpCFG(Operation *op, CFGContext &context,
                      llvm::raw_ostream &os) {
  CFGPrinter(os, context).dumpGraph(op);
}
