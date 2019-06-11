#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <vector>

template <typename T>
void radix_sort(T begin, T end) {
    const size_t n_bits = sizeof(T) * 8;
    for (int i = 0; i < n_bits; i++) {
        auto it =
            std::stable_partition(begin, end, [&i](const auto &elem) -> bool {
                return (elem & (1 << i)) == 0;
            });
        std::copy(begin, end, std::ostream_iterator<size_t>(std::cout, " "));
        std::cout << '\n';
    }
}

int main() {
    size_t n = 10;
    std::vector<size_t> arr(n);
    srand(time(NULL));
    std::generate(arr.begin(), arr.end(),
                  []() -> size_t { return rand() % 10; });
    radix_sort(arr.begin(), arr.end());
    std::copy(arr.begin(), arr.end(),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << '\n';
    return 0;
}