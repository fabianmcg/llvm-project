add_lldb_library(lldbValueObject
  DILAST.cpp
  DILEval.cpp
  DILLexer.cpp
  DILParser.cpp
  ValueObject.cpp
  ValueObjectCast.cpp
  ValueObjectChild.cpp
  ValueObjectConstResult.cpp
  ValueObjectConstResultCast.cpp
  ValueObjectConstResultChild.cpp
  ValueObjectConstResultImpl.cpp
  ValueObjectDynamicValue.cpp
  ValueObjectList.cpp
  ValueObjectMemory.cpp
  ValueObjectRegister.cpp
  ValueObjectSyntheticFilter.cpp
  ValueObjectUpdater.cpp
  ValueObjectVariable.cpp
  ValueObjectVTable.cpp

  LINK_COMPONENTS
    Support
    Demangle
    TargetParser
  LINK_LIBS
    lldbBreakpoint
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
  )
