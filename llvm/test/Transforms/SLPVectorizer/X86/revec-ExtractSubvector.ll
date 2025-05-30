; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=slp-vectorizer -S -mtriple=x86_64-unknown-linux-gnu -slp-revec -pass-remarks-output=%t < %s | FileCheck %s
; RUN: FileCheck --input-file=%t --check-prefix=YAML %s

; See https://reviews.llvm.org/D70068 and https://reviews.llvm.org/D70587 for context

; YAML: --- !Passed
; YAML: Pass:            slp-vectorizer
; YAML: Name:            VectorizedList
; YAML: Function:        StructOfVectors
; YAML: Args:
; YAML:   - String:          'SLP vectorized with cost '
; YAML:   - Cost:            '-10'
; YAML:   - String:          ' and with tree size '
; YAML:   - TreeSize:        '3'

; YAML: --- !Missed
; YAML: Pass:            slp-vectorizer
; YAML: Name:            NotBeneficial
; YAML: Function:        StructOfVectors
; YAML: Args:
; YAML:   - String:          'List vectorization was possible but not beneficial with cost '
; YAML:   - Cost:            '0'
; YAML:   - String:          ' >= '
; YAML:   - Treshold:        '0'

; Checks that vector insertvalues into the struct become SLP seeds.
define { <2 x float>, <2 x float> } @StructOfVectors(ptr %Ptr) {
; CHECK-LABEL: @StructOfVectors(
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr [[PTR:%.*]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = fadd fast <4 x float> [[TMP1]], <float 1.100000e+01, float 1.200000e+01, float 1.300000e+01, float 1.400000e+01>
; CHECK-NEXT:    [[TMP6:%.*]] = shufflevector <4 x float> [[TMP2]], <4 x float> poison, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT:    [[TMP7:%.*]] = shufflevector <4 x float> [[TMP2]], <4 x float> poison, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT:    [[RET0:%.*]] = insertvalue { <2 x float>, <2 x float> } poison, <2 x float> [[TMP6]], 0
; CHECK-NEXT:    [[RET1:%.*]] = insertvalue { <2 x float>, <2 x float> } [[RET0]], <2 x float> [[TMP7]], 1
; CHECK-NEXT:    ret { <2 x float>, <2 x float> } [[RET1]]
;
  %L0 = load float, ptr %Ptr
  %GEP1 = getelementptr inbounds float, ptr %Ptr, i64 1
  %L1 = load float, ptr %GEP1
  %GEP2 = getelementptr inbounds float, ptr %Ptr, i64 2
  %L2 = load float, ptr %GEP2
  %GEP3 = getelementptr inbounds float, ptr %Ptr, i64 3
  %L3 = load float, ptr %GEP3

  %Fadd0 = fadd fast float %L0, 1.1e+01
  %Fadd1 = fadd fast float %L1, 1.2e+01
  %Fadd2 = fadd fast float %L2, 1.3e+01
  %Fadd3 = fadd fast float %L3, 1.4e+01

  %VecIn0 = insertelement <2 x float> poison, float %Fadd0, i64 0
  %VecIn1 = insertelement <2 x float> %VecIn0, float %Fadd1, i64 1

  %VecIn2 = insertelement <2 x float> poison, float %Fadd2, i64 0
  %VecIn3 = insertelement <2 x float> %VecIn2, float %Fadd3, i64 1

  %Ret0 = insertvalue {<2 x float>, <2 x float>} poison, <2 x float> %VecIn1, 0
  %Ret1 = insertvalue {<2 x float>, <2 x float>} %Ret0, <2 x float> %VecIn3, 1
  ret {<2 x float>, <2 x float>} %Ret1
}
