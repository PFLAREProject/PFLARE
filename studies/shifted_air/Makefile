# Build inside the CI image: make (uses PETSC_DIR/PETSC_ARCH and the PFLARE lib in ../../lib)
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
PFLARE_DIR := $(CURDIR)/../..
CFLAGS += -I$(PFLARE_DIR)/include -Wall -Werror
shift_study: shift_study.o
	${CLINKER} -o $@ $^ -Wl,-rpath,$(PFLARE_DIR)/lib -L$(PFLARE_DIR)/lib -lpflare ${PETSC_LIB}
