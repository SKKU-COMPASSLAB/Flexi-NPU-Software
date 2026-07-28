#ifndef __LIBFLEXI_FLEXI_HX_H
#define __LIBFLEXI_FLEXI_HX_H

#include <random>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#include "riscv/extension.h"
#include "riscv/rocc.h"
#include "riscv/mmu.h"
#include "riscv/trap.h"

#include "flexi_blas.h"

typedef uint8_t  byte_t;
#define XCUSTOM_HX 2

class flexi_hx_t;

#define MAKECUSTOMFN(opcode) custom ## opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

class flexi_hx_t : public extension_t
{
public:
  // Constructors
  flexi_hx_t();
  ~flexi_hx_t();

  // RoCC Custom Instruction
  reg_t CUSTOMFN(XCUSTOM_HX)(rocc_insn_t insn, reg_t xs1, reg_t xs2);

  // Methods: Data Manipulation (READ / WRITE)
  void read_buffer_from_dram(reg_t addr, byte_t *buffer, const size_t buffer_size);
  void write_buffer_to_dram (reg_t addr, byte_t *buffer, const size_t buffer_size);

  // Methods: Spike Interface for Custom RoCC Commands
  void HX_FWHT(reg_t rs1, reg_t rs2);
  void HX_SYNC(reg_t rs1, reg_t rs2);

  // Methods: Common Interface for RoCC Extension in Spike
  const char* name() const { return "flexi_hx"; }
  void reset() {};
  void set_processor(processor_t* p) { this->p = p; }
  void user(reg_t dram_addr, reg_t sp_addr) {}

  std::vector<insn_desc_t> get_instructions();
  std::vector<insn_desc_t> get_instructions(const processor_t &p);
  std::vector<disasm_insn_t*> get_disasms();
  std::vector<disasm_insn_t*> get_disasms(const processor_t *p);

private:
  // Attributes: Common Variables for RoCC Extension in Spike
  reg_t cause;
  reg_t aux;
  processor_t* p;
 
  const unsigned user_funct = 2;
  const unsigned HX_FWHT_funct = 1;
  const unsigned HX_SYNC_funct = 0;
};

#endif  // __LIBFLEXI_FLEXI_HX_H