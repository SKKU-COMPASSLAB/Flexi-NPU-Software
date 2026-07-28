#include "flexi_hx.h"
#include <time.h>

using namespace std;

REGISTER_EXTENSION(flexi_hx, []()
                   { return new flexi_hx_t; })

/******************************************************
 * Methods: Data Manipulation
 ******************************************************/

flexi_hx_t::flexi_hx_t() : cause(0), aux(0), p(nullptr) {}
flexi_hx_t::~flexi_hx_t() {}

void flexi_hx_t::read_buffer_from_dram(reg_t addr, byte_t *buffer, const size_t buffer_size)
{
  for (size_t byte_idx = 0; byte_idx < buffer_size; byte_idx++)
  {
    buffer[byte_idx] = p->get_mmu()->load<byte_t>(addr + byte_idx);
  }
}

void flexi_hx_t::write_buffer_to_dram(reg_t addr, byte_t *buffer, const size_t buffer_size)
{
  for (size_t byte_idx = 0; byte_idx < buffer_size; byte_idx++)
  {
    p->get_mmu()->store<byte_t>(addr + byte_idx, buffer[byte_idx]);
  }
}

/******************************************************
 * Methods: Spike Interface for Custom RoCC Commands
 ******************************************************/
static const uint16_t HX_FWHT_SCALE_TAB[15] = {
  0x3F80, 0x3F35, 0x3F00, 0x3EB5, 0x3E80, 0x3E35, 0x3E00, 0x3DB5,
  0x3D80, 0x3D35, 0x3D00, 0x3CB5, 0x3C80, 0x3C35, 0x3C00
};

void flexi_hx_t::HX_FWHT(reg_t rs1, reg_t rs2)
{
  constexpr reg_t SV39_ADDR_MASK = (reg_t{1} << 39) - 1;
  constexpr reg_t SV39_SIGN_BIT = reg_t{1} << 38;
  constexpr reg_t RESERVED_BIT_39 = reg_t{1} << 39;
  constexpr reg_t RESERVED_BITS_63_60 = reg_t{0xF} << 60;

  auto const src = rs1;

  reg_t dst = rs2 & SV39_ADDR_MASK;
  if (dst & SV39_SIGN_BIT)
    dst |= ~SV39_ADDR_MASK;

  auto const logN = (rs2 >> 40) & reg_t{0xF};
  auto const numVecs = (rs2 >> 44) & reg_t{0xFFFF};
  if (p == nullptr) {
    fprintf(stderr, "(FLEXI_HX) processor is not initialized\n");
    return;
  }
  if ((rs2 & (RESERVED_BIT_39 | RESERVED_BITS_63_60)) != 0) {
    fprintf(stderr, "(FLEXI_HX) reserved rs2 bits must be zero\n");
    return;
  }
  if (logN >= sizeof(HX_FWHT_SCALE_TAB) / sizeof(HX_FWHT_SCALE_TAB[0])) {
    fprintf(stderr, "(FLEXI_HX) unsupported logN=%lu\n", logN);
    return;
  }
  if (numVecs == 0) {
    fprintf(stderr, "(FLEXI_HX) numVecs must be nonzero\n");
    return;
  }
  if ((src & reg_t{0x7}) != 0 || (dst & reg_t{0x7}) != 0) {
    fprintf(stderr, "(FLEXI_HX) src and dst must be 8-byte aligned\n");
    return;
  }

  auto const numelPerVec = 1 << logN;
  auto const packed_buffer_size = numelPerVec * 2;  // 2 bytes per element (bfloat16)
  auto const scale =
    fnblas_dtype_bf16_bits_to_fp32(HX_FWHT_SCALE_TAB[logN]);

  byte_t *src_packed_buffer = new byte_t[packed_buffer_size];
  byte_t *dst_packed_buffer = new byte_t[packed_buffer_size];

  fnblas_vector_t dst_vec = FNBLAS_VECTOR_INITIALIZER;
  fnblas_vector_create(&dst_vec, numelPerVec, BF16);

  for (int v = 0; v < numVecs; v++) {
    auto const src_vec_addr = src + v * packed_buffer_size;
    auto const dst_vec_addr = dst + v * packed_buffer_size;

    read_buffer_from_dram(src_vec_addr, src_packed_buffer, packed_buffer_size);
    fnblas_vector_initialize_from_packed_buffer(
      &dst_vec,
      reinterpret_cast<const byte_t*>(src_packed_buffer)
    );
    
    for (int dd = 1; dd < numelPerVec; dd <<= 1) {
      for (int i = 0; i < numelPerVec; i++) {
        if ((i & dd) == 0) {
          auto const j = i | dd;
          float a, b;
          fnblas_vector_at(
            &dst_vec, i, reinterpret_cast<byte_t*>(&a)
          );
          fnblas_vector_at(
            &dst_vec, j, reinterpret_cast<byte_t*>(&b)
          );
          
          auto const dst_i = fnblas_dtype_softfloat_bf16_to_fp32(a + b);
          auto const dst_j = fnblas_dtype_softfloat_bf16_to_fp32(a - b);
          fnblas_vector_set(
            &dst_vec, i, reinterpret_cast<const byte_t*>(&dst_i)
          );
          fnblas_vector_set(
            &dst_vec, j, reinterpret_cast<const byte_t*>(&dst_j)
          );
        }
      }
    }

    for (int i = 0; i < numelPerVec; ++i) {
      float value;
      fnblas_vector_at(
        &dst_vec, i, reinterpret_cast<byte_t*>(&value)
      );
      value = fnblas_dtype_softfloat_bf16_to_fp32(value * scale);
      fnblas_vector_set(
        &dst_vec, i, reinterpret_cast<const byte_t*>(&value)
      );
    }

    fnblas_vector_get_packed_buffer(
      &dst_vec,
      reinterpret_cast<byte_t*>(dst_packed_buffer)
    );
    write_buffer_to_dram(dst_vec_addr, dst_packed_buffer, packed_buffer_size);
  }

  fnblas_vector_destroy(&dst_vec);
  delete[] src_packed_buffer;
  delete[] dst_packed_buffer;
}

void flexi_hx_t::HX_SYNC(reg_t rs1, reg_t rs2)
{
  // synchronization is NOP for ISS
}

reg_t flexi_hx_t::CUSTOMFN(XCUSTOM_HX)(rocc_insn_t insn, reg_t xs1, reg_t xs2)
{
  auto const c_opcode = insn.funct;

  if (c_opcode == HX_FWHT_funct)
    HX_FWHT(xs1, xs2);
  else if (c_opcode == HX_SYNC_funct)
    HX_SYNC(xs1, xs2);
  else
  {
    fprintf(stderr, "(FLEXI_HX) unknown instruction with rs1=%16lx rs2=%16lx\n", xs1, xs2);
  }

  return 0;
}

static reg_t flexi_hx_custom2(processor_t* p, insn_t insn, reg_t pc)
{
  auto* rocc = static_cast<flexi_hx_t*>(p->get_extension("flexi_hx"));
  rocc_insn_union_t decoded;
  state_t* state = p->get_state();
  decoded.i = insn;
  const reg_t xs1 = decoded.r.xs1 ? state->XPR[insn.rs1()] : -1;
  const reg_t xs2 = decoded.r.xs2 ? state->XPR[insn.rs2()] : -1;

  rocc->set_processor(p);
  const reg_t xd = rocc->custom2(decoded.r, xs1, xs2);
  if (decoded.r.xd) {
    state->log_reg_write[insn.rd() << 4] = {xd, 0};
    state->XPR.write(insn.rd(), xd);
  }
  return pc + 4;
}

/******************************************************
 * Methods: Common Interface for RoCC Extension in Spike
 ******************************************************/

std::vector<insn_desc_t> flexi_hx_t::get_instructions()
{
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE2, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC, flexi_hx_custom2);
  return insns;
}

std::vector<disasm_insn_t *> flexi_hx_t::get_disasms()
{
  std::vector<disasm_insn_t *> insns;
  return insns;
}

std::vector<insn_desc_t> flexi_hx_t::get_instructions(const processor_t &p)
{
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE2, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC, flexi_hx_custom2);
  return insns;
}

std::vector<disasm_insn_t *> flexi_hx_t::get_disasms(const processor_t *p)
{
  std::vector<disasm_insn_t *> insns;
  return insns;
}
