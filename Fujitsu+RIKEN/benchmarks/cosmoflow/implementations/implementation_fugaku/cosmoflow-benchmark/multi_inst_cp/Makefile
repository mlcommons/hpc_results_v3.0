LANG_MODULE := lang/tcsds-1.2.33

SHELL    := /bin/bash
CXX      := mpiFCCpx
CXXFLAGS := -Nclang -std=c++11
LDLIBS   := -lcrypto

multi_inst_cp: multi_inst_cp_bcast.cpp
	module unload lang; \
	module load $(LANG_MODULE); \
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

clean:
	$(RM) multi_inst_cp
