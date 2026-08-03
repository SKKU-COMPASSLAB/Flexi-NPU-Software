#include "flexi.h"
#include <time.h>

using namespace std;

REGISTER_EXTENSION(flexi, []()
                   { return new flexi_t; })

/******************************************************
 * Methods: Data Manipulation
 ******************************************************/

flexi_t::flexi_t() : cause(0), aux(0), p(nullptr) {
  onc_mem = (byte_t *)calloc(FLEXI_ONC_MEM_SIZE, sizeof(byte_t));
  mxu_wgt_tile = FNBLAS_MATRIX_INITIALIZER;
  mxu_psum_tile = FNBLAS_MATRIX_INITIALIZER;
}

flexi_t::~flexi_t() {
  free(onc_mem);
  if (is_mxu_wgt_preloaded) {
    fnblas_matrix_destroy(&mxu_wgt_tile);
  }
  if (is_mxu_psum_preloaded) {
    fnblas_matrix_destroy(&mxu_psum_tile);
  }
}

/******************************************************
 * Methods: Data Manipulation
 ******************************************************/

void flexi_t::read_buffer_from_dram(reg_t addr, byte_t *buffer, const size_t buffer_size)
{
  for (size_t byte_idx = 0; byte_idx < buffer_size; byte_idx++)
  {
    buffer[byte_idx] = p->get_mmu()->load<byte_t>(addr + byte_idx);
  }
}

void flexi_t::write_buffer_to_dram(reg_t addr, byte_t *buffer, const size_t buffer_size)
{
  for (size_t byte_idx = 0; byte_idx < buffer_size; byte_idx++)
  {
    p->get_mmu()->store<byte_t>(addr + byte_idx, buffer[byte_idx]);
  }
}

/******************************************************
 * Methods: Spike Interface for Custom RoCC Commands
 ******************************************************/

void flexi_t::GEMMPreload(reg_t preload_rs1, reg_t preload_rs2)
{
  auto const addr = (preload_rs1 >> 38) & 0x3FFFFFF;
  auto const row_numel = (preload_rs2 >> 7) & 0x3FF;
  auto const total_row = (preload_rs2 >> 17) & 0x3FF;
  auto const mat_type = (preload_rs2 >> 27) & 0x7;
  auto const dtype = (preload_rs2 >> 38) & 0x7;
  auto const opcode = preload_rs2 & 0x3F;

  auto const fnblas_dtype = static_cast<fnblas_dtype_t>(dtype);
  byte_t *tile_buffer = onc_mem + addr;
  auto const row_buffer_size = row_numel * fnblas_dtype_size_of(fnblas_dtype) / fnblas_dtype_pack_size_of(fnblas_dtype);

  if (mat_type == MAT_A) {
    fprintf(stderr, "flexi_t::GEMMPreload: Cannot preload matrix A to the MXU\n");
    return;
  } else if (mat_type == MAT_B) {
    if (is_mxu_wgt_preloaded) {
      fnblas_matrix_destroy(&mxu_wgt_tile);
    }

    fnblas_matrix_create(&mxu_wgt_tile, FLEXI_MXU_N_TILE_SIZE, FLEXI_MXU_K_TILE_SIZE, fnblas_dtype);
    byte_t *mxu_wgt_packed_buffer = (byte_t *)calloc(fnblas_matrix_packed_buffer_size(&mxu_wgt_tile), sizeof(byte_t));
    auto const row_stride_size = FLEXI_MXU_K_TILE_SIZE * fnblas_dtype_size_of(fnblas_dtype) / fnblas_dtype_pack_size_of(fnblas_dtype);

    for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
      memcpy(mxu_wgt_packed_buffer + row_idx * row_stride_size, tile_buffer + row_idx * (row_buffer_size + 0), row_buffer_size);
    }

    fnblas_matrix_initialize_from_packed_buffer(&mxu_wgt_tile, mxu_wgt_packed_buffer);
    free(mxu_wgt_packed_buffer);

    is_mxu_wgt_preloaded = true;
  } else if (mat_type == MAT_C) {
    if (is_mxu_psum_preloaded) {
      fnblas_matrix_destroy(&mxu_psum_tile);
    }

    fnblas_matrix_create(&mxu_psum_tile, FLEXI_MXU_M_TILE_SIZE, FLEXI_MXU_N_TILE_SIZE, fnblas_dtype);
    byte_t *mxu_psum_packed_buffer = (byte_t *)calloc(fnblas_matrix_packed_buffer_size(&mxu_psum_tile), sizeof(byte_t));
    auto const row_stride_size = FLEXI_MXU_N_TILE_SIZE * fnblas_dtype_size_of(fnblas_dtype) / fnblas_dtype_pack_size_of(fnblas_dtype);

    for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
      memcpy(mxu_psum_packed_buffer + row_idx * row_stride_size, tile_buffer + row_idx * (row_buffer_size + 0), row_buffer_size);
    }

    fnblas_matrix_initialize_from_packed_buffer(&mxu_psum_tile, mxu_psum_packed_buffer);
    free(mxu_psum_packed_buffer);

    is_mxu_psum_preloaded = true;
  } else {
    fprintf(stderr, "flexi_t::GEMMPreload: Invalid matrix type %ld\n", mat_type);
  }
}

void flexi_t::GEMMExecuteS1(reg_t execute_rs1, reg_t execute_rs2)
{
  static bool fp4_diagnostic_printed = false;
  auto const addr = (execute_rs1 >> 38) & 0x3FFFFFF;
  auto const row_numel = (execute_rs2 >> 7) & 0x3FF;
  auto const total_row = (execute_rs2 >> 17) & 0x3FF;
  auto const mat_type = (execute_rs2 >> 27) & 0x7;
  auto const dtype = (execute_rs2 >> 38) & 0x7;
  auto const res_dtype = (execute_rs2 >> 30) & 0x7F;
  auto const opcode = execute_rs2 & 0x3F;

  auto const fnblas_ifm_dtype = static_cast<fnblas_dtype_t>(dtype);
  auto const fnblas_res_dtype = static_cast<fnblas_dtype_t>(res_dtype);
  auto const ifm_tile_size = row_numel * total_row * fnblas_dtype_size_of(fnblas_ifm_dtype) / fnblas_dtype_pack_size_of(fnblas_ifm_dtype);

  fnblas_matrix_t mxu_ifm_tile = FNBLAS_MATRIX_INITIALIZER;
  fnblas_matrix_t mxu_res_tile = FNBLAS_MATRIX_INITIALIZER;

  if (mat_type == MAT_A) {
    byte_t *ifm_tile_buffer = onc_mem + addr;
    fnblas_matrix_create(&mxu_ifm_tile, FLEXI_MXU_M_TILE_SIZE, FLEXI_MXU_K_TILE_SIZE, static_cast<fnblas_dtype_t>(dtype));
    byte_t *mxu_ifm_packed_buffer = (byte_t *)calloc(fnblas_matrix_packed_buffer_size(&mxu_ifm_tile), sizeof(byte_t));
    auto const row_stride_size = FLEXI_MXU_K_TILE_SIZE * fnblas_dtype_size_of(fnblas_ifm_dtype) / fnblas_dtype_pack_size_of(fnblas_ifm_dtype);
    auto const row_buffer_size = row_numel * fnblas_dtype_size_of(fnblas_ifm_dtype) / fnblas_dtype_pack_size_of(fnblas_ifm_dtype);

    for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
      memcpy(mxu_ifm_packed_buffer + row_idx * row_stride_size, ifm_tile_buffer + row_idx * (row_buffer_size + 0), row_buffer_size);
    }

    fnblas_matrix_initialize_from_packed_buffer(&mxu_ifm_tile, mxu_ifm_packed_buffer);
    free(mxu_ifm_packed_buffer);

    if (!fp4_diagnostic_printed && dtype == FP4 && getenv("FLEXI_DEBUG_FP4") != nullptr) {
      auto const *ifm = reinterpret_cast<const float *>(mxu_ifm_tile._buffer);
      auto const *wgt = reinterpret_cast<const float *>(mxu_wgt_tile._buffer);
      auto const *psum = reinterpret_cast<const float *>(mxu_psum_tile._buffer);
      fprintf(stderr, "[FP4-DIAG] A[5,0:8]=");
      for (size_t index = 0; index < 8; index++) fprintf(stderr, " %.1f", ifm[5 * FLEXI_MXU_K_TILE_SIZE + index]);
      fprintf(stderr, "\n[FP4-DIAG] B[0,0:8]=");
      for (size_t index = 0; index < 8; index++) fprintf(stderr, " %.1f", wgt[index]);
      fprintf(stderr, "\n[FP4-DIAG] C[5,0]=%.8g\n", psum[5 * FLEXI_MXU_N_TILE_SIZE]);
    }
  } else if (mat_type == MAT_B) {
    fprintf(stderr, "flexi_t::GEMMExecuteS1: Cannot execute matrix B on the MXU\n");
    return;
  } else if (mat_type == MAT_C) {
    fprintf(stderr, "flexi_t::GEMMExecuteS1: Cannot execute matrix C on the MXU\n");
    return;
  } else {
    fprintf(stderr, "flexi_t::GEMMExecuteS1: Invalid matrix type %ld\n", mat_type);
    return;
  }

  auto error = fnblas_matrix_create(
    &mxu_res_tile,
    FLEXI_MXU_M_TILE_SIZE,
    FLEXI_MXU_N_TILE_SIZE,
    static_cast<fnblas_dtype_t>(res_dtype)
  );
  if (error == FNBLAS_SUCCESS) {
    error = fnblas_op_mm_matmul(
      &mxu_res_tile, &mxu_ifm_tile, &mxu_wgt_tile
    );
  }

  if (error == FNBLAS_SUCCESS && is_mxu_psum_preloaded) {
    fnblas_matrix_t accumulated_tile = FNBLAS_MATRIX_INITIALIZER;
    error = fnblas_op_me_add(
      &accumulated_tile, &mxu_res_tile, &mxu_psum_tile
    );
    if (error == FNBLAS_SUCCESS) {
      fnblas_matrix_destroy(&mxu_psum_tile);
      mxu_psum_tile = accumulated_tile;
      accumulated_tile = FNBLAS_MATRIX_INITIALIZER;
    }
    fnblas_matrix_destroy(&accumulated_tile);
  } else if (error == FNBLAS_SUCCESS) {
    mxu_psum_tile = mxu_res_tile;
    mxu_res_tile = FNBLAS_MATRIX_INITIALIZER;
    is_mxu_psum_preloaded = true;
  }

  if (!fp4_diagnostic_printed && dtype == FP4 && getenv("FLEXI_DEBUG_FP4") != nullptr && error == FNBLAS_SUCCESS) {
    auto const *result = reinterpret_cast<const float *>(mxu_psum_tile._buffer);
    fprintf(stderr, "[FP4-DIAG] D[5,0]=%.8g\n", result[5 * FLEXI_MXU_N_TILE_SIZE]);
    fp4_diagnostic_printed = true;
  }

  if (error != FNBLAS_SUCCESS) {
    fprintf(
      stderr,
      "flexi_t::GEMMExecuteS1: GEMM failed (error=%d)\n",
      static_cast<int>(error)
    );
  }

  fnblas_matrix_destroy(&mxu_ifm_tile);
  fnblas_matrix_destroy(&mxu_res_tile);
}

void flexi_t::GEMMFlush(reg_t flush_rs1, reg_t flush_rs2)
{
  auto const addr = (flush_rs1 >> 38) & 0x3FFFFFF;
  auto const row_numel = (flush_rs2 >> 7) & 0x3FF;
  auto const total_row = (flush_rs2 >> 17) & 0x3FF;
  auto const mat_type = (flush_rs2 >> 27) & 0x7;
  auto const dtype = (flush_rs2 >> 38) & 0x7;
  auto const opcode = flush_rs2 & 0x3F;

  if (mat_type == MAT_D) {
    if (!is_mxu_psum_preloaded) {
      fprintf(stderr, "flexi_t::GEMMFlush: No preloaded matrix C to flush\n");
      return;
    }
    byte_t *psum_packed_buffer = (byte_t *)malloc(fnblas_matrix_packed_buffer_size(&mxu_psum_tile));
    fnblas_matrix_get_packed_buffer(&mxu_psum_tile, psum_packed_buffer);

    auto const fnblas_dtype = static_cast<fnblas_dtype_t>(dtype);
    auto const row_stride_size = FLEXI_MXU_N_TILE_SIZE * fnblas_dtype_size_of(fnblas_dtype) / fnblas_dtype_pack_size_of(fnblas_dtype);
    auto const row_buffer_size = row_numel * fnblas_dtype_size_of(fnblas_dtype) / fnblas_dtype_pack_size_of(fnblas_dtype);

    for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
      memcpy(onc_mem + addr + row_idx * (row_buffer_size + 0), psum_packed_buffer + row_idx * row_stride_size, row_buffer_size);
    }

    free(psum_packed_buffer);
    fnblas_matrix_destroy(&mxu_psum_tile);
    is_mxu_psum_preloaded = false;
  } else {
    fprintf(stderr, "flexi_t::GEMMFlush: Invalid matrix type %ld for flushing\n", mat_type);
  }
}

void flexi_t::DMALoad(reg_t load_rs1, reg_t load_rs2)
{
  static size_t fp4_dma_diagnostic_count = 0;
  auto const off_addr = ((load_rs1 & 0x3FFFFFFFFF) << 10) | ((load_rs2 >> 54) & 0x3FF);
  auto const onc_addr = (load_rs1 >> 38) & 0x3FFFFFF;
  auto       row_numel = (load_rs2 >> 7) & 0x3FF;
  auto       total_row = (load_rs2 >> 17) & 0x3FF;
  auto const stride = (load_rs2 >> 27) & 0x1FFF;
  auto const trans = (load_rs2 >> 40) & 0x1;
  auto const dtype = (load_rs2 >> 41) & 0x7;
  auto const zero_pad = (load_rs2 >> 44) & 0x3FF;
  auto const opcode = load_rs2 & 0x3F;

  if (dtype == FP4 && getenv("FLEXI_DEBUG_FP4") != nullptr && fp4_dma_diagnostic_count < 20) {
    fprintf(stderr, "[FP4-DIAG] DMA[%zu] dram=0x%lx onc=0x%lx rows=%lu cols=%lu stride=%lu pad=%lu\n", fp4_dma_diagnostic_count, off_addr, onc_addr, total_row, row_numel, stride, zero_pad);
    fp4_dma_diagnostic_count++;
  }

  auto fnblas_dtype = static_cast<fnblas_dtype_t>(dtype);
  auto dtype_size = fnblas_dtype_size_of(fnblas_dtype);
  auto dtype_pack_size = fnblas_dtype_pack_size_of(fnblas_dtype);
  auto stride_size = (stride * dtype_size + dtype_pack_size - 1) / dtype_pack_size;
  auto row_size = (row_numel * dtype_size + dtype_pack_size - 1) / dtype_pack_size;
  auto padded_row_size = ((row_numel + zero_pad) * dtype_size + dtype_pack_size - 1) / dtype_pack_size;
  auto zero_pad_size = padded_row_size - row_size;

  byte_t *dma_buffer = (byte_t *)malloc(row_size * total_row);
  byte_t *onc_buffer = onc_mem + onc_addr;

  for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
    read_buffer_from_dram(off_addr + row_idx * stride_size, dma_buffer + row_idx * row_size, row_size);
  }

  if (trans) {
    fnblas_matrix_t original_matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t transposed_matrix = FNBLAS_MATRIX_INITIALIZER;

    fnblas_matrix_create(&original_matrix, total_row, row_numel, static_cast<fnblas_dtype_t>(dtype));
    fnblas_matrix_initialize_from_packed_buffer(&original_matrix, dma_buffer);
    fnblas_op_mi_transpose(&transposed_matrix, &original_matrix);
    fnblas_matrix_get_packed_buffer(&transposed_matrix, dma_buffer);
    fnblas_matrix_destroy(&original_matrix);
    fnblas_matrix_destroy(&transposed_matrix);

    auto temp = row_numel;
    row_numel = total_row;
    total_row = temp;

    row_size = row_numel * dtype_size / dtype_pack_size;
    padded_row_size = (row_numel + zero_pad) * dtype_size / dtype_pack_size;
    zero_pad_size = padded_row_size - row_size;
  }

  for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
    memcpy(onc_buffer + row_idx * padded_row_size, dma_buffer + row_idx * row_size, row_size);
    memset(onc_buffer + row_idx * padded_row_size + row_size, 0, zero_pad_size);
  }

  free(dma_buffer);
}

void flexi_t::DMAStore(reg_t store_rs1, reg_t store_rs2)
{
  auto const off_addr = ((store_rs1 & 0x3FFFFFFFFF) << 10) | ((store_rs2 >> 54) & 0x3FF);
  auto const onc_addr = (store_rs1 >> 38) & 0x3FFFFFF;
  auto       row_numel = (store_rs2 >> 7) & 0x3FF;    // zero pad included
  auto       total_row = (store_rs2 >> 17) & 0x3FF;
  auto const stride = (store_rs2 >> 27) & 0x1FFF;
  auto const trans = (store_rs2 >> 40) & 0x1;
  auto const dtype = (store_rs2 >> 41) & 0x7;
  auto const zero_pad = (store_rs2 >> 44) & 0x3FF;
  auto const opcode = store_rs2 & 0x3F;

  if (row_numel < zero_pad) {
    fprintf(stderr, "flexi_t::DMAStore: Invalid configuration: row_numel (%lu) < zero_pad (%lu)\n", row_numel, zero_pad);
    return;
  }
  if (row_numel == zero_pad) {
    return; // nothing to store
  }

  auto fnblas_dtype = static_cast<fnblas_dtype_t>(dtype);
  auto dtype_size = fnblas_dtype_size_of(fnblas_dtype);
  auto dtype_pack_size = fnblas_dtype_pack_size_of(fnblas_dtype);
  auto stride_size = (stride * dtype_size + dtype_pack_size - 1) / dtype_pack_size;
  auto padded_row_size = (row_numel * dtype_size + dtype_pack_size - 1) / dtype_pack_size;
  auto row_size = ((row_numel - zero_pad) * dtype_size + dtype_pack_size - 1) / dtype_pack_size;
  
  byte_t *dma_buffer = (byte_t *)malloc(row_size * total_row);
  byte_t *onc_buffer = onc_mem + onc_addr;

  for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
    memcpy(dma_buffer + row_idx * row_size, onc_buffer + row_idx * padded_row_size, row_size);
  }

  if (trans) {
    fnblas_matrix_t transposed_matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t restored_matrix = FNBLAS_MATRIX_INITIALIZER;

    fnblas_matrix_create(&transposed_matrix, total_row, row_numel - zero_pad, fnblas_dtype);
    fnblas_matrix_initialize_from_packed_buffer(&transposed_matrix, dma_buffer);
    fnblas_op_mi_transpose(&restored_matrix, &transposed_matrix);
    fnblas_matrix_get_packed_buffer(&restored_matrix, dma_buffer);
    fnblas_matrix_destroy(&transposed_matrix);
    fnblas_matrix_destroy(&restored_matrix);

    auto temp = row_numel;
    row_numel = total_row;
    total_row = temp - zero_pad;

    row_size = row_numel * dtype_size / dtype_pack_size;
  }

  for (size_t row_idx = 0; row_idx < total_row; row_idx++) {
    write_buffer_to_dram(off_addr + row_idx * stride_size, dma_buffer + row_idx * row_size, row_size);
  }

  free(dma_buffer);
}

reg_t flexi_t::CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2)
{
  auto const c_opcode = xs2 & 0x3F;

  if (c_opcode == 3)
    GEMMPreload(xs1, xs2);
  else if (c_opcode == 5)
    GEMMExecuteS1(xs1, xs2);
  else if (c_opcode == 7)
    GEMMFlush(xs1, xs2);
  else if (c_opcode == 12)
    DMALoad(xs1, xs2);
  else if (c_opcode == 13)
    DMAStore(xs1, xs2);
  else if (c_opcode == 0) {
    // this instruction is FlexiSync, which is a no-op in simulation framework
  }
  else
  {
    fprintf(stderr, "unknown instruction with rs1=%16lx rs2=%16lx\n", xs1, xs2);
  }

  return 0;
}

static reg_t flexi_custom3(processor_t* p, insn_t insn, reg_t pc)
{
  auto* rocc = static_cast<flexi_t*>(p->get_extension("flexi"));
  rocc_insn_union_t decoded;
  state_t* state = p->get_state();
  decoded.i = insn;
  const reg_t xs1 = decoded.r.xs1 ? state->XPR[insn.rs1()] : -1;
  const reg_t xs2 = decoded.r.xs2 ? state->XPR[insn.rs2()] : -1;

  rocc->set_processor(p);
  const reg_t xd = rocc->custom3(decoded.r, xs1, xs2);
  if (decoded.r.xd) {
    state->log_reg_write[insn.rd() << 4] = {xd, 0};
    state->XPR.write(insn.rd(), xd);
  }
  return pc + 4;
}

/******************************************************
 * Methods: Common Interface for RoCC Extension in Spike
 ******************************************************/

std::vector<insn_desc_t> flexi_t::get_instructions()
{
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE3, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC, flexi_custom3);
  return insns;
}

std::vector<disasm_insn_t *> flexi_t::get_disasms()
{
  std::vector<disasm_insn_t *> insns;
  return insns;
}

std::vector<insn_desc_t> flexi_t::get_instructions(const processor_t &p)
{
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE3, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC, flexi_custom3);
  return insns;
}

std::vector<disasm_insn_t *> flexi_t::get_disasms(const processor_t *p)
{
  std::vector<disasm_insn_t *> insns;
  return insns;
}
