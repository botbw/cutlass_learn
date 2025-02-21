// nvcc -I${CUTLASS_REPO_PATH}/include layout_tensor_core.cu && ./a.out && rm a.out
#include "cute/atom/mma_traits.hpp"

#include "cute/arch/mma_sm70.hpp"
#include "cute/atom/mma_traits_sm70.hpp"

#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_traits_sm80.hpp"

#include "cute/arch/mma_sm90.hpp"
#include "cute/atom/mma_traits_sm90.hpp"

#include "cute/util/print.hpp"

#include <stdio.h>

using namespace cute;

int main() {
    using TRAITED_SM70_8x8x4_F32F16F16F32_NT = MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>;
    printf("\n\n\nSM70_8x8x4_F32F16F16F32_NT A matrix layout:\n");
    print_layout(TRAITED_SM70_8x8x4_F32F16F16F32_NT::ALayout{});
    printf("\n\n\nSM70_8x8x4_F32F16F16F32_NT B matrix layout:\n");
    print_layout(TRAITED_SM70_8x8x4_F32F16F16F32_NT::BLayout{});
    printf("\n\n\nSM70_8x8x4_F32F16F16F32_NT C matrix layout:\n");
    print_layout(TRAITED_SM70_8x8x4_F32F16F16F32_NT::CLayout{});

    using TRAITED_SM80_16x8x8_F32F16F16F32_TN = MMA_Traits<SM80_16x8x8_F32F16F16F32_TN>;
    printf("\n\n\nSM80_16x8x8_F32F16F16F32_TN A matrix layout:\n");
    print_layout(TRAITED_SM80_16x8x8_F32F16F16F32_TN::ALayout{});
    printf("\n\n\nSM80_16x8x8_F32F16F16F32_TN B matrix layout:\n");
    print_layout(TRAITED_SM80_16x8x8_F32F16F16F32_TN::BLayout{});
    printf("\n\n\nSM80_16x8x8_F32F16F16F32_TN C matrix layout:\n");
    print_layout(TRAITED_SM80_16x8x8_F32F16F16F32_TN::CLayout{});    

    return 0;
}