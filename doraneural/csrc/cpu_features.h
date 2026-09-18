#ifndef DORANEURAL_CPU_FEATURES_H
#define DORANEURAL_CPU_FEATURES_H

#include <cstdio>
#include <cstring>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DORANEURAL_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DORANEURAL_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
#define DORANEURAL_ARCH_ARM32 1
#else
#define DORANEURAL_ARCH_UNKNOWN 1
#endif

#if defined(__linux__) && (defined(DORANEURAL_ARCH_ARM64) || defined(DORANEURAL_ARCH_ARM32))
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

struct CPUFeatures {
    const char* arch_name = "Unknown";
    bool has_avx512f = false;
    bool has_avx512vnni = false;
    bool has_avx512bw = false;
    bool has_avx2 = false;
    bool has_fma = false;
    bool has_f16c = false;
    bool has_arm_neon = false;
    bool has_arm_dotprod = false;
    bool has_arm_fp16 = false;
    const char* optimal_backend = "SCALAR";
};

inline CPUFeatures detect_cpu_features() {
    static CPUFeatures feat;
    static bool detected = false;
    if (detected) return feat;

#if defined(DORANEURAL_ARCH_X86)
    feat.arch_name = "x86_64";
    __builtin_cpu_init();
#ifdef __AVX512F__
    feat.has_avx512f = __builtin_cpu_supports("avx512f") > 0;
#endif
#ifdef __AVX512VNNI__
    feat.has_avx512vnni = __builtin_cpu_supports("avx512vnni") > 0;
#endif
#ifdef __AVX512BW__
    feat.has_avx512bw = __builtin_cpu_supports("avx512bw") > 0;
#endif
#ifdef __AVX2__
    feat.has_avx2 = __builtin_cpu_supports("avx2") > 0;
#endif
#ifdef __FMA__
    feat.has_fma = __builtin_cpu_supports("fma") > 0;
#endif
#ifdef __F16C__
    feat.has_f16c = true;
#endif

    if (feat.has_avx512vnni && feat.has_avx512bw) {
        feat.optimal_backend = "AVX512_VNNI";
    } else if (feat.has_avx2) {
        feat.optimal_backend = "AVX2_FMA";
    } else {
        feat.optimal_backend = "SCALAR";
    }

#elif defined(DORANEURAL_ARCH_ARM64)
    feat.arch_name = "ARM64";
#if defined(__ARM_NEON) || defined(__aarch64__)
    feat.has_arm_neon = true;
#endif
#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
#ifdef HWCAP_ASIMD
    feat.has_arm_neon = (hwcap & HWCAP_ASIMD) != 0;
#endif
#ifdef HWCAP_ASIMDDP
    feat.has_arm_dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
#endif
#ifdef HWCAP_FPHP
    feat.has_arm_fp16 = (hwcap & HWCAP_FPHP) != 0;
#endif
#endif // __linux__

#if defined(__ARM_FEATURE_DOTPROD)
    feat.has_arm_dotprod = true;
#endif

    if (feat.has_arm_dotprod) {
        feat.optimal_backend = "ARM_NEON_DOTPROD";
    } else if (feat.has_arm_neon) {
        feat.optimal_backend = "ARM_NEON";
    } else {
        feat.optimal_backend = "SCALAR";
    }

#else
    feat.arch_name = "Generic";
    feat.optimal_backend = "SCALAR";
#endif

    detected = true;
    return feat;
}

inline void print_cpu_features() {
    CPUFeatures feat = detect_cpu_features();
    std::fprintf(stderr, "[doraneural CPU] Architecture: %s | Optimal Backend: %s\n", feat.arch_name, feat.optimal_backend);
    std::fprintf(stderr, "  SIMD Capabilities: ");
    if (feat.has_avx512vnni) std::fprintf(stderr, "AVX-512_VNNI ");
    if (feat.has_avx512f) std::fprintf(stderr, "AVX-512F ");
    if (feat.has_avx2) std::fprintf(stderr, "AVX2 ");
    if (feat.has_arm_neon) std::fprintf(stderr, "ARM_NEON ");
    if (feat.has_arm_dotprod) std::fprintf(stderr, "ARM_DOTPROD(asimddp) ");
    if (feat.has_arm_fp16) std::fprintf(stderr, "ARM_FP16 ");
    std::fprintf(stderr, "\n");
}

#endif // DORANEURAL_CPU_FEATURES_H
