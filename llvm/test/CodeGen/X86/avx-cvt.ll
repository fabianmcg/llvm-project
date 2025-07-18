; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: llc < %s -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512

define <8 x float> @sitofp00(<8 x i32> %a) nounwind {
; CHECK-LABEL: sitofp00:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtdq2ps %ymm0, %ymm0
; CHECK-NEXT:    retq
  %b = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %b
}

define <8 x i32> @fptosi00(<8 x float> %a) nounwind {
; CHECK-LABEL: fptosi00:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvttps2dq %ymm0, %ymm0
; CHECK-NEXT:    retq
  %b = fptosi <8 x float> %a to <8 x i32>
  ret <8 x i32> %b
}

define <4 x double> @sitofp01(<4 x i32> %a) {
; CHECK-LABEL: sitofp01:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtdq2pd %xmm0, %ymm0
; CHECK-NEXT:    retq
  %b = sitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %b
}

define <8 x float> @sitofp02(<8 x i16> %a) {
; AVX-LABEL: sitofp02:
; AVX:       # %bb.0:
; AVX-NEXT:    vpmovsxwd %xmm0, %xmm1
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; AVX-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX-NEXT:    retq
;
; AVX512-LABEL: sitofp02:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX512-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX512-NEXT:    retq
  %b = sitofp <8 x i16> %a to <8 x float>
  ret <8 x float> %b
}

define <4 x i32> @fptosi01(<4 x double> %a) {
; CHECK-LABEL: fptosi01:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvttpd2dq %ymm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %b = fptosi <4 x double> %a to <4 x i32>
  ret <4 x i32> %b
}

define <8 x float> @fptrunc00(<8 x double> %b) nounwind {
; AVX-LABEL: fptrunc00:
; AVX:       # %bb.0:
; AVX-NEXT:    vcvtpd2ps %ymm0, %xmm0
; AVX-NEXT:    vcvtpd2ps %ymm1, %xmm1
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    retq
;
; AVX512-LABEL: fptrunc00:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vcvtpd2ps %zmm0, %ymm0
; AVX512-NEXT:    retq
  %a = fptrunc <8 x double> %b to <8 x float>
  ret <8 x float> %a
}

define <4 x float> @fptrunc01(<2 x double> %a0, <4 x float> %a1) nounwind {
; CHECK-LABEL: fptrunc01:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtsd2ss %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %ext = extractelement <2 x double> %a0, i32 0
  %cvt = fptrunc double %ext to float
  %res = insertelement <4 x float> %a1, float %cvt, i32 0
  ret <4 x float> %res
}

define <4 x double> @fpext00(<4 x float> %b) nounwind {
; CHECK-LABEL: fpext00:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtps2pd %xmm0, %ymm0
; CHECK-NEXT:    retq
  %a = fpext <4 x float> %b to <4 x double>
  ret <4 x double> %a
}

define <2 x double> @fpext01(<2 x double> %a0, <4 x float> %a1) nounwind {
; CHECK-LABEL: fpext01:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtss2sd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %ext = extractelement <4 x float> %a1, i32 0
  %cvt = fpext float %ext to double
  %res = insertelement <2 x double> %a0, double %cvt, i32 0
  ret <2 x double> %res
}

define double @funcA(ptr nocapture %e) nounwind uwtable readonly ssp {
; CHECK-LABEL: funcA:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtsi2sdq (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %tmp1 = load i64, ptr %e, align 8
  %conv = sitofp i64 %tmp1 to double
  ret double %conv
}

define double @funcB(ptr nocapture %e) nounwind uwtable readonly ssp {
; CHECK-LABEL: funcB:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtsi2sdl (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %tmp1 = load i32, ptr %e, align 4
  %conv = sitofp i32 %tmp1 to double
  ret double %conv
}

define float @funcC(ptr nocapture %e) nounwind uwtable readonly ssp {
; CHECK-LABEL: funcC:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtsi2ssl (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %tmp1 = load i32, ptr %e, align 4
  %conv = sitofp i32 %tmp1 to float
  ret float %conv
}

define float @funcD(ptr nocapture %e) nounwind uwtable readonly ssp {
; CHECK-LABEL: funcD:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vcvtsi2ssq (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %tmp1 = load i64, ptr %e, align 8
  %conv = sitofp i64 %tmp1 to float
  ret float %conv
}

define void @fpext() nounwind uwtable {
; CHECK-LABEL: fpext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT:    vcvtss2sd %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    vmovsd %xmm0, -{{[0-9]+}}(%rsp)
; CHECK-NEXT:    retq
  %f = alloca float, align 4
  %d = alloca double, align 8
  %tmp = load float, ptr %f, align 4
  %conv = fpext float %tmp to double
  store double %conv, ptr %d, align 8
  ret void
}

define double @nearbyint_f64(double %a) {
; CHECK-LABEL: nearbyint_f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundsd $12, %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call double @llvm.nearbyint.f64(double %a)
  ret double %res
}
declare double @llvm.nearbyint.f64(double %p)

define float @floor_f32(float %a) {
; CHECK-LABEL: floor_f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundss $9, %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %res = call float @llvm.floor.f32(float %a)
  ret float %res
}
declare float @llvm.floor.f32(float %p)

define float @floor_f32_load(ptr %aptr) optsize {
; CHECK-LABEL: floor_f32_load:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundss $9, (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %a = load float, ptr %aptr
  %res = call float @llvm.floor.f32(float %a)
  ret float %res
}

define float @floor_f32_load_pgso(ptr %aptr) !prof !14 {
; CHECK-LABEL: floor_f32_load_pgso:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundss $9, (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %a = load float, ptr %aptr
  %res = call float @llvm.floor.f32(float %a)
  ret float %res
}

define double @nearbyint_f64_load(ptr %aptr) optsize {
; CHECK-LABEL: nearbyint_f64_load:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundsd $12, (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %a = load double, ptr %aptr
  %res = call double @llvm.nearbyint.f64(double %a)
  ret double %res
}

define double @nearbyint_f64_load_pgso(ptr %aptr) !prof !14 {
; CHECK-LABEL: nearbyint_f64_load_pgso:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundsd $12, (%rdi), %xmm15, %xmm0
; CHECK-NEXT:    retq
  %a = load double, ptr %aptr
  %res = call double @llvm.nearbyint.f64(double %a)
  ret double %res
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
