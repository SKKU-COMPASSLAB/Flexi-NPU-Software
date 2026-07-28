#include "flexi_tensor.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(float actual, float expected, const char* message)
{
    if (std::fabs(actual - expected) > 1.0e-6f)
        throw std::runtime_error(message);
}

float vector_at(const fnblas_vector_t* vector, std::size_t index)
{
    float value = 0.0f;
    require(fnblas_vector_at(vector, index, reinterpret_cast<byte_t*>(&value)) == FNBLAS_SUCCESS, "vector at");
    return value;
}

float matrix_at(const fnblas_matrix_t* matrix, std::size_t row, std::size_t col)
{
    float value = 0.0f;
    require(fnblas_matrix_at(matrix, row, col, reinterpret_cast<byte_t*>(&value)) == FNBLAS_SUCCESS, "matrix at");
    return value;
}

float tensor_at(const flexi_tensor_t* tensor, flexi_tuple_t index)
{
    float value = 0.0f;
    require(flexi_tensor_at(tensor, index, reinterpret_cast<byte_t*>(&value)) == FLEXI_TENSOR_SUCCESS, "tensor at");
    return value;
}

void test_vector_view()
{
    float buffer[6] = {0, 1, 2, 3, 4, 5};
    float value = 9.0f;
    fnblas_vector_t view = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_vector_create_view(&view, reinterpret_cast<byte_t*>(buffer), 6, FP32) == FNBLAS_SUCCESS, "vector view create");
    require(view._buffer == reinterpret_cast<byte_t*>(buffer), "vector view is not zero-copy");
    require(fnblas_vector_set(&view, 2, reinterpret_cast<const byte_t*>(&value)) == FNBLAS_SUCCESS, "vector view set");
    require_close(vector_at(&view, 2), 9.0f, "vector view read");
    require_close(buffer[2], 9.0f, "vector view write was not reflected");
    require(fnblas_vector_destroy(&view) == FNBLAS_SUCCESS, "vector view destroy");
    require_close(buffer[2], 9.0f, "vector view destroy modified the source");
}

void test_matrix_views()
{
    float buffer[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    float value = 17.0f;
    fnblas_matrix_t matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t reshaped = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t invalid = FNBLAS_MATRIX_INITIALIZER;
    fnblas_vector_t row = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_matrix_create_view(&matrix, reinterpret_cast<byte_t*>(buffer), 3, 4, FP32) == FNBLAS_SUCCESS, "matrix view create");
    require(matrix._buffer == reinterpret_cast<byte_t*>(buffer), "matrix view is not zero-copy");
    require(fnblas_matrix_create_row_vector_view(&row, &matrix, 1) == FNBLAS_SUCCESS, "matrix row view create");
    require(row._buffer == reinterpret_cast<byte_t*>(&buffer[4]), "matrix row view offset");
    require(fnblas_vector_set(&row, 2, reinterpret_cast<const byte_t*>(&value)) == FNBLAS_SUCCESS, "matrix row view set");
    require_close(matrix_at(&matrix, 1, 2), 17.0f, "matrix row write was not reflected");
    require(fnblas_matrix_create_reshape_view(&reshaped, &matrix, 2, 6) == FNBLAS_SUCCESS, "matrix reshape view create");
    require(reshaped._buffer == matrix._buffer, "matrix reshape view is not zero-copy");
    require_close(matrix_at(&reshaped, 1, 0), 17.0f, "matrix reshape layout");
    require(fnblas_matrix_create_reshape_view(&invalid, &matrix, 5, 5) == FNBLAS_ERR_MISMATCH, "matrix reshape mismatch");
    require(fnblas_vector_destroy(&row) == FNBLAS_SUCCESS, "matrix row view destroy");
    require(fnblas_matrix_destroy(&reshaped) == FNBLAS_SUCCESS, "matrix reshape view destroy");
    require(fnblas_matrix_destroy(&matrix) == FNBLAS_SUCCESS, "matrix view destroy");
    require_close(buffer[6], 17.0f, "matrix view destroy modified the source");
}

void test_tensor_views()
{
    float buffer[120];
    const flexi_tuple_t shape = FLEXI_TUPLE(2, 3, 4, 5);
    const flexi_tuple_t reshape = FLEXI_TUPLE(6, 20);
    const flexi_tuple_t invalid_shape = FLEXI_TUPLE(7, 20);
    const flexi_tuple_t matrix_index = FLEXI_TUPLE(1, 2);
    const flexi_tuple_t vector_index = FLEXI_TUPLE(1, 2, 3);
    float value = 33.0f;
    flexi_tensor_t tensor = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t reshaped = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t invalid = FLEXI_TENSOR_INITIALIZER;
    fnblas_matrix_t matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_vector_t vector = FNBLAS_VECTOR_INITIALIZER;
    for (std::size_t index = 0; index < 120; ++index)
        buffer[index] = static_cast<float>(index);
    require(flexi_tensor_create_view(&tensor, reinterpret_cast<byte_t*>(buffer), shape, FP32) == FLEXI_TENSOR_SUCCESS, "tensor view create");
    require(tensor._buffer == reinterpret_cast<byte_t*>(buffer), "tensor view is not zero-copy");
    require(tensor._strides[0] == 60 && tensor._strides[1] == 20 && tensor._strides[2] == 5 && tensor._strides[3] == 1, "tensor view strides");
    require(flexi_tensor_create_matrix_view(&matrix, &tensor, matrix_index) == FLEXI_TENSOR_SUCCESS, "tensor matrix view create");
    require(matrix._buffer == reinterpret_cast<byte_t*>(&buffer[100]), "tensor matrix view offset");
    require(matrix._n_rows == 4 && matrix._n_cols == 5, "tensor matrix view shape");
    require_close(matrix_at(&matrix, 3, 4), 119.0f, "tensor matrix view layout");
    require(flexi_tensor_create_vector_view(&vector, &tensor, vector_index) == FLEXI_TENSOR_SUCCESS, "tensor vector view create");
    require(vector._buffer == reinterpret_cast<byte_t*>(&buffer[115]), "tensor vector view offset");
    require(vector._n_elements == 5, "tensor vector view shape");
    require(fnblas_vector_set(&vector, 4, reinterpret_cast<const byte_t*>(&value)) == FNBLAS_SUCCESS, "tensor vector view set");
    require_close(vector_at(&vector, 4), 33.0f, "tensor vector view read");
    require_close(tensor_at(&tensor, FLEXI_TUPLE(1, 2, 3, 4)), 33.0f, "tensor vector write was not reflected");
    require(flexi_tensor_create_reshape_view(&reshaped, &tensor, reshape) == FLEXI_TENSOR_SUCCESS, "tensor reshape view create");
    require(reshaped._buffer == tensor._buffer, "tensor reshape view is not zero-copy");
    require_close(tensor_at(&reshaped, FLEXI_TUPLE(5, 19)), 33.0f, "tensor reshape layout");
    require(flexi_tensor_create_reshape_view(&invalid, &tensor, invalid_shape) == FLEXI_TENSOR_ERR_MISMATCH, "tensor reshape mismatch");
    require(fnblas_matrix_destroy(&matrix) == FNBLAS_SUCCESS, "tensor matrix view destroy");
    require(fnblas_vector_destroy(&vector) == FNBLAS_SUCCESS, "tensor vector view destroy");
    require(flexi_tensor_destroy(&reshaped) == FLEXI_TENSOR_SUCCESS, "tensor reshape view destroy");
    require(flexi_tensor_destroy(&tensor) == FLEXI_TENSOR_SUCCESS, "tensor view destroy");
    require_close(buffer[119], 33.0f, "tensor view destroy modified the source");
}

void test_tensor_create()
{
    flexi_tensor_t tensor = FLEXI_TENSOR_INITIALIZER;
    float value = 7.0f;
    require(flexi_tensor_create(&tensor, FLEXI_TUPLE(2, 3), FP32) == FLEXI_TENSOR_SUCCESS, "tensor create");
    require(tensor._n_dims == 2 && tensor._shape[0] == 2 && tensor._shape[1] == 3, "tensor create shape");
    require(flexi_tensor_set(&tensor, FLEXI_TUPLE(1, 2), reinterpret_cast<const byte_t*>(&value)) == FLEXI_TENSOR_SUCCESS, "tensor tuple set");
    require_close(tensor_at(&tensor, FLEXI_TUPLE(1, 2)), 7.0f, "tensor tuple at");
    require(flexi_tensor_at(&tensor, FLEXI_TUPLE(1), reinterpret_cast<byte_t*>(&value)) != FLEXI_TENSOR_SUCCESS, "tensor index rank mismatch");
    require(flexi_tensor_destroy(&tensor) == FLEXI_TENSOR_SUCCESS, "tensor destroy");
}

} // namespace

int main()
{
    try {
        test_vector_view();
        test_matrix_views();
        test_tensor_views();
        test_tensor_create();
        std::cout << "t5_tensor_view: all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "t5_tensor_view: FAILED: " << error.what() << '\n';
        return 1;
    }
}
