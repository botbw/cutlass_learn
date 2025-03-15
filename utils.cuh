#ifndef UTILS_CUH
#define UTILS_CUH

template <typename T>
void randn(T *arr, int size, double vmin, double vmax) {
    for (int i = 0; i < size; i++) {
        arr[i] = int(vmin + (vmax - vmin) * rand() / RAND_MAX);
    }
}

#endif // UTILS_CUH