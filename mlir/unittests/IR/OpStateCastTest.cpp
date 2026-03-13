//===- OpStateCastTest.cpp - Test OpState value-type casting --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that isa/dyn_cast between OpState-derived value types (e.g.,
// OpInterface -> ConcreteOp) correctly route through Operation* for the TypeID
// check.
//
// Without the CastInfo specialization in OpDefinition.h, the default CastInfo
// for value types returns a reference from castFailed(), which is undefined
// behavior. The compiler can exploit this UB to eliminate the isPossible
// (classof/TypeID) check entirely, causing dyn_cast to unconditionally
// "succeed" -- even when the underlying operation has a completely different
// TypeID.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// Test: dyn_cast from OpInterface to a non-matching ConcreteOp must fail.
//===----------------------------------------------------------------------===//

TEST(OpStateCastTest, DynCastInterfaceToWrongConcreteOpFails) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(module->getBody(), module->getBody()->begin());

  // Create a SideEffectOp -- it implements MemoryEffectOpInterface.
  auto sideEffectOp = test::SideEffectOp::create(builder,
                                                  builder.getUnknownLoc(),
                                                  builder.getI32Type());

  // Get the interface value (an OpState-derived value type, NOT Operation*).
  MemoryEffectOpInterface iface = cast<MemoryEffectOpInterface>(sideEffectOp);
  ASSERT_TRUE(iface);

  // Sanity: casting the interface back to the correct concrete op succeeds.
  EXPECT_TRUE(isa<test::SideEffectOp>(iface));
  auto backCast = dyn_cast<test::SideEffectOp>(iface);
  EXPECT_TRUE(backCast);

  // The critical test: casting the interface to a WRONG concrete op must fail.
  // Without the CastInfo fix, this dyn_cast "succeeds" due to UB at -O2.
  EXPECT_FALSE(isa<test::SymbolOp>(iface));
  auto wrongCast = dyn_cast<test::SymbolOp>(iface);
  EXPECT_FALSE(wrongCast);
}

TEST(OpStateCastTest, DynCastConcreteOpToWrongConcreteOpFails) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(module->getBody(), module->getBody()->begin());

  auto sideEffectOp = test::SideEffectOp::create(builder,
                                                  builder.getUnknownLoc(),
                                                  builder.getI32Type());

  // Cast from SideEffectOp (value type) to a wrong op type.
  EXPECT_FALSE(isa<test::SymbolOp>(sideEffectOp));
  auto wrongCast = dyn_cast<test::SymbolOp>(sideEffectOp);
  EXPECT_FALSE(wrongCast);

  // Same cast through Operation* should also fail (sanity).
  EXPECT_FALSE(isa<test::SymbolOp>(sideEffectOp.getOperation()));
}

TEST(OpStateCastTest, DynCastInterfaceToCorrectConcreteOpSucceeds) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(module->getBody(), module->getBody()->begin());

  auto sideEffectOp = test::SideEffectOp::create(builder,
                                                  builder.getUnknownLoc(),
                                                  builder.getI32Type());
  MemoryEffectOpInterface iface = cast<MemoryEffectOpInterface>(sideEffectOp);

  // Cast from interface to the correct concrete op should succeed.
  EXPECT_TRUE(isa<test::SideEffectOp>(iface));
  auto correctCast = dyn_cast<test::SideEffectOp>(iface);
  ASSERT_TRUE(correctCast);

  // The result should refer to the same operation.
  EXPECT_EQ(correctCast.getOperation(), sideEffectOp.getOperation());
}

TEST(OpStateCastTest, IsaConsistentBetweenOpStateAndOperationPtr) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(module->getBody(), module->getBody()->begin());

  auto sideEffectOp = test::SideEffectOp::create(builder,
                                                  builder.getUnknownLoc(),
                                                  builder.getI32Type());
  MemoryEffectOpInterface iface = cast<MemoryEffectOpInterface>(sideEffectOp);
  Operation *op = sideEffectOp.getOperation();

  // Both paths must agree for a matching type.
  EXPECT_EQ(isa<test::SideEffectOp>(iface), isa<test::SideEffectOp>(op));

  // Both paths must agree for a non-matching type.
  EXPECT_EQ(isa<test::SymbolOp>(iface), isa<test::SymbolOp>(op));

  // Both paths must agree for ModuleOp (definitely wrong).
  EXPECT_EQ(isa<ModuleOp>(iface), isa<ModuleOp>(op));
}
