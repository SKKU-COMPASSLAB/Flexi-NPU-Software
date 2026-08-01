#include "flexi_blas.h"

#include <cerrno>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require_close(float actual, float expected, const char* message)
{
    if (std::fabs(actual - expected) > 1.0e-6f)
        throw std::runtime_error(message);
}

fnblas_matrix_t make_matrix(
    std::size_t rows,
    std::size_t cols,
    const std::vector<float>& values)
{
    fnblas_matrix_t matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_create(&matrix, rows, cols, FP32);
    fnblas_matrix_initialize_from_unpacked_buffer(
        &matrix,
        reinterpret_cast<const byte_t*>(values.data())
    );
    return matrix;
}

float at(const fnblas_matrix_t* matrix, std::size_t row, std::size_t col)
{
    float value = 0.0f;
    fnblas_matrix_at(
        matrix, row, col, reinterpret_cast<byte_t*>(&value)
    );
    return value;
}

} // namespace

int main()
{
    try {
        fnblas_matrix_t lhs = make_matrix(2, 3, {1, 2, 3, 4, 5, 6});
        fnblas_matrix_t rhs = make_matrix(2, 3, {6, 5, 4, 3, 2, 1});
        fnblas_matrix_t sum = FNBLAS_MATRIX_INITIALIZER;
        fnblas_matrix_create(&sum, 2, 3, FP32);
        byte_t* const sum_buffer = sum._buffer;
        fnblas_op_me_add(&sum, &lhs, &rhs);
        if (sum._buffer != sum_buffer)
            throw std::runtime_error("matrix result was reallocated");
        for (std::size_t row = 0; row < 2; ++row)
            for (std::size_t col = 0; col < 3; ++col)
                require_close(at(&sum, row, col), 7.0f, "matrix add");

        if (fnblas_matrix_create(&sum, 4, 4, FP32) !=
                FNBLAS_ERR_UNKNOWN ||
            sum._buffer != sum_buffer)
            throw std::runtime_error("duplicate matrix create");

        fnblas_matrix_t wrong_result = FNBLAS_MATRIX_INITIALIZER;
        fnblas_matrix_create(&wrong_result, 1, 1, FP32);
        byte_t* const wrong_buffer = wrong_result._buffer;
        if (fnblas_op_me_add(&wrong_result, &lhs, &rhs) !=
                FNBLAS_ERR_MISMATCH ||
            wrong_result._buffer != wrong_buffer)
            throw std::runtime_error("matrix result shape validation");

        fnblas_matrix_t transpose = FNBLAS_MATRIX_INITIALIZER;
        fnblas_op_mi_transpose(&transpose, &lhs);
        require_close(at(&transpose, 0, 1), 4.0f, "matrix transpose");
        require_close(at(&transpose, 2, 1), 6.0f, "matrix transpose");

        fnblas_matrix_t rhs_t = make_matrix(2, 3, {7, 8, 9, 10, 11, 12});
        fnblas_matrix_t product = FNBLAS_MATRIX_INITIALIZER;
        fnblas_op_mm_matmul(&product, &lhs, &rhs_t);
        require_close(at(&product, 0, 0), 50.0f, "matrix matmul");
        require_close(at(&product, 0, 1), 68.0f, "matrix matmul");
        require_close(at(&product, 1, 0), 122.0f, "matrix matmul");
        require_close(at(&product, 1, 1), 167.0f, "matrix matmul");

        fnblas_matrix_t mixed_product = FNBLAS_MATRIX_INITIALIZER;
        fnblas_matrix_create(&mixed_product, 2, 2, BF16);
        if (fnblas_op_mm_matmul(&mixed_product, &lhs, &rhs_t) != FNBLAS_SUCCESS || mixed_product._dtype != BF16)
            throw std::runtime_error("matrix matmul must preserve explicit floating-point output dtype");
        require_close(at(&mixed_product, 1, 1), 167.0f, "mixed dtype matrix matmul");

        fnblas_matrix_t integer_product = FNBLAS_MATRIX_INITIALIZER;
        fnblas_matrix_create(&integer_product, 2, 2, INT32);
        if (fnblas_op_mm_matmul(&integer_product, &lhs, &rhs_t) != FNBLAS_ERR_MISMATCH)
            throw std::runtime_error("matrix matmul must reject FP-to-INT output mismatch");

        fnblas_matrix_t mixed_elementwise = FNBLAS_MATRIX_INITIALIZER;
        fnblas_matrix_create(&mixed_elementwise, 2, 3, BF16);
        if (fnblas_op_me_add(&mixed_elementwise, &lhs, &rhs) != FNBLAS_ERR_MISMATCH)
            throw std::runtime_error("matrix elementwise operation must require matching output dtype");

        std::vector<byte_t> packed(6 * sizeof(float));
        fnblas_matrix_get_packed_buffer(&lhs, packed.data());
        fnblas_matrix_t restored = FNBLAS_MATRIX_INITIALIZER;
        fnblas_matrix_create(&restored, 2, 3, FP32);
        fnblas_matrix_initialize_from_packed_buffer(
            &restored, packed.data()
        );
        require_close(at(&restored, 1, 2), 6.0f, "matrix packed round trip");

        fnblas_matrix_destroy(&lhs);
        fnblas_matrix_destroy(&rhs);
        fnblas_matrix_destroy(&sum);
        fnblas_matrix_destroy(&wrong_result);
        fnblas_matrix_destroy(&transpose);
        fnblas_matrix_destroy(&rhs_t);
        fnblas_matrix_destroy(&product);
        fnblas_matrix_destroy(&mixed_product);
        fnblas_matrix_destroy(&integer_product);
        fnblas_matrix_destroy(&mixed_elementwise);
        fnblas_matrix_destroy(&restored);
        std::cout << "t4_blas_matrix_test: all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "t4_blas_matrix_test: FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
