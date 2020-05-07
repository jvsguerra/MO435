# COMMANDS
MKDIR=mkdir -p
CP=cp
PYTHON=python
RM=rm -r -f --verbose

# FOLDERS
SRC_FOLDER=src
BUILD_FOLDER=report
BIN_FOLDER=bin
ASSETS_FOLDER=input
OUTPUT_FOLDER=output
REQ=requirements.txt

# SRCs and OBJs
MAIN=$(SRC_FOLDER)/main.py

# BUILD COMMAND
build:
	$(MKDIR) $(OUTPUT_FOLDER)
	$(MKDIR) $(OUTPUT_FOLDER)/functions
	$(MKDIR) $(OUTPUT_FOLDER)/train_size
	$(MKDIR) $(OUTPUT_FOLDER)/test_size
	$(MKDIR) $(OUTPUT_FOLDER)/kernelSE
	$(MKDIR) $(OUTPUT_FOLDER)/kernelSE/ell
	$(MKDIR) $(OUTPUT_FOLDER)/kernelSE/s
	$(MKDIR) $(OUTPUT_FOLDER)/kernelSE/noise
	$(MKDIR) $(OUTPUT_FOLDER)/kernels/
	$(MKDIR) $(OUTPUT_FOLDER)/kernels/prior
	$(MKDIR) $(OUTPUT_FOLDER)/kernels/prior/noise-free
	$(MKDIR) $(OUTPUT_FOLDER)/kernels/prior/noisy
	$(MKDIR) $(OUTPUT_FOLDER)/kernels/
	$(MKDIR) $(OUTPUT_FOLDER)/kernels/posterior
	$(MKDIR) $(OUTPUT_FOLDER)/GP2D

exec:
	pip3 install -r $(SRC_FOLDER)/$(REQ)
	python3 $(MAIN)

runall: build exec

# CLEAN COMMAND
clean:
	$(RM) $(BIN_FOLDER)
	$(RM) $(OUTPUT_FOLDER)
