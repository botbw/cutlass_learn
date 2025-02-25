// nvcc -I${CUTLASS_REPO_PATH}/include layout_algebra.cu && ./a.out && rm a.out
#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include "cute/tensor.hpp"
#include "cutlass/layout/matrix.h"

#include <stdio.h>

using namespace cute;

template <typename Layout>
void print_1d_layout(Layout layout) {
    print(layout);
    puts("");
    for (int i = 0; i < size(layout); i++) {
        print(layout(i));
        printf(" ");
    }
}

int main() {
    // Coalesce
    {
        auto a = Layout<
                    Shape<  Shape<_2, _1>,Shape <_1,_6>>,
                    Stride<Stride<_1, _2>,Stride<_6,_2>>
                >{};
        print(a); puts("");
        print(coalesce(a)); puts("");
    }

    // By-mode Coalesce
    {
        auto a = Layout<
                    Shape<  Shape<_2, _1>,Shape <_1,_6>>,
                    Stride<Stride<_1, _2>,Stride<_6,_2>>
                >{};
        print(a); puts("");
        print(coalesce(a, Step<_1>{})); puts("");
        print(coalesce(a, Step<_1, _1>{})); puts("");
    }

    // Composition
    // 1-D composition examples
    {
        auto a = Layout<
            Shape<_8>,
            Stride<_3>
        >{};
        print_1d_layout(a); puts(""); puts("");
        {
            auto b = Layout<
                        Shape<_3>,
                        Stride<_1>
                    >{};
            auto res = composition(a, b);
            print_1d_layout(b); puts("");
            print_1d_layout(res); puts(""); puts("");
        }
        {
            auto b = Layout<
                        Shape<_3>,
                        Stride<_2>
                    >{};
            auto res = composition(a, b);
            print_1d_layout(b); puts("");
            print_1d_layout(res); puts(""); puts("");
        }
        {
            auto b = Layout<
                        Shape<_3>,
                        Stride<_3>
                    >{};
            auto res = composition(a, b);
            print_1d_layout(b); puts("");
            print_1d_layout(res); puts(""); puts("");
        }
        {
            auto b = Layout<
                        Shape<_8>,
                        Stride<_2>
                    >{};
            auto res = composition(a, b);
            print_1d_layout(b); puts("");
            print_1d_layout(res); puts(""); puts("");
            print(a(_8{}));
        }
    }

    // N-D composition
    {
        {
            auto a = Layout<
                        Shape<_3, _4>,
                        Stride<_4, _1>
                    >{};
            auto b = Layout<
                        _4,
                        _3
                    >{};
            auto res = composition(a, b);
            print_1d_layout(a); puts(""); puts("");
            print_1d_layout(b); puts(""); puts("");
            print_1d_layout(res);
        }
        {
            auto a = Layout<
                        Shape<_4, _5>,
                        Stride<_5, _3>
                    >{};
            auto b = Layout<
                        Shape<_2, _2>,
                        Stride<_2, _1>
                    >{};
            auto res = composition(a, b);
            print_1d_layout(a); puts(""); puts("");
            print_1d_layout(b); puts(""); puts("");
            print_1d_layout(res);
        }
    }

    // By-mode composition
    {
        auto a = Layout<
                    Shape<_4, _4>,
                    Stride<_3, _4>
                >{};
        {
            auto tiler = make_tile(
                Layout<Shape<_2>, Stride<_1>>{},
                Layout<Shape<_2>, Stride<_1>>{}  // Shape<_2, _2>{};
            );
            auto res = composition(a, tiler);
            print_layout(a); puts("");
            print_layout(res); puts("");
        }
        {
            auto tiler = make_tile(
                Layout<Shape<_2>, Stride<_2>>{},
                Layout<Shape<_2>, Stride<_1>>{}
            );
            auto res = composition(a, tiler);
            print_layout(a); puts("");
            print_layout(res); puts("");
        }
        {
            auto tiler = make_tile(
                Layout<Shape<_4>, Stride<_1>>{},
                Layout<Shape<_2>, Stride<_2>>{}
            );
            auto res = composition(a, tiler);
            print_layout(a); puts("");
            print_layout(res); puts("");
        }
    }

    // Concatenation
    {
        Layout a = Layout<_3,_2>{};
        Layout b = Layout<_4, _1>{};
        Layout ab = append(a, b);
        print_1d_layout(a); puts("");
        print_1d_layout(b); puts("");
        print_layout(ab); puts("");
    }
    {
        Layout a = Layout<
            Shape<_3,_2>,
            Stride<_1, _3>
        >{};
        Layout b = Layout<_4, _2>{};
        Layout ab = append(a, b);
        print_layout(a); puts("");
        print_1d_layout(b); puts("");
        for (int i = 0; i < get<2>(ab.shape()); i++) {
            for (int j = 0; j < get<0>(ab.shape()); j++) {
                for (int k = 0; k < get<1>(ab.shape()); k++) {
                    printf("%d ", ab(make_coord(j, k, i)));
                }
                puts("");
            }
            puts("");
        }
    }
    
    // Complement
    {
        auto a = Layout<
                    Shape<_8>,
                    Stride<_3>
                >{};
        auto b = Layout<
                    Shape<_2>,
                    Stride<_1>
                >{};
        auto res = complement(b, size(a));
        print_1d_layout(a); puts("");
        print_1d_layout(b); puts("");
        print_1d_layout(res); puts("");
        print_1d_layout(composition(a, b)); puts("");
        print_1d_layout(res); puts("");
        print_1d_layout(composition(a, res)); puts("");
        print_1d_layout(append(b, res)); puts("");
        print_1d_layout(composition(a, append(b, res))); puts("");
    }

    // product
    {
        auto a = Layout<
                    Shape<_2, _3>,
                    Stride<_3, _1>
                >{};
        auto b = Layout<
                    Shape<_2>,
                    Stride<_2>
                >{};
        auto res = logical_product(a, b);
        print_layout(a); puts("");
        print_1d_layout(b); puts("");
        print_1d_layout(append(a, complement(a, size(a) * cosize(b)))); puts("");
        print_1d_layout(res); puts("");
    }
    {
        auto a0 = Layout<
                    Shape<_2>,
                    Stride<_5>
                >{};
        auto b0 = Layout<
                    Shape<_3>,
                    Stride<_5>
                >{};
        auto res0 = coalesce(logical_product(a0, b0));
        print_1d_layout(a0); puts("");
        print_1d_layout(b0); puts("");
        print_1d_layout(append(a0, complement(a0, size(a0) * cosize(b0)))); puts("");
        print_1d_layout(res0); puts("");

        auto a1 = Layout<
                    Shape<_5>,
                    Stride<_1>
                >{};
        auto b1 = Layout<
                    Shape<_4>,
                    Stride<_6>
                >{};
        auto res1 = coalesce(logical_product(a1, b1));
        print_1d_layout(a1); puts("");
        print_1d_layout(b1); puts("");
        print_1d_layout(append(a1, complement(a1, size(a1) * cosize(b1)))); puts("");
        print_1d_layout(res1); puts("");

        print_layout(append(res0, res1)); puts("");
    }
    {
        auto a = Layout<
                    Shape<_4, _5>,
                    Stride<_1, _4>
                >{};
        auto b = Layout<
                    Shape<_2, _3>,
                    Stride<_3, _1>
                >{};
        print_layout(a); puts("");
        print_layout(b); puts("");
        auto logical_result = logical_product(a, b);
        print_layout(logical_result);
        auto blocked_result = blocked_product(a, b);
        print_layout(blocked_result); puts("");
        print_layout(
            coalesce(
                zip(
                    get<0>(logical_result),
                    get<1>(logical_result)
                ),
                Step<_1, _1>{}
            )
        );
    }
    return 0;
}