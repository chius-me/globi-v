# Makefile for camera_capture — standalone Milk-V Duo camera snapshot tool
# Cross-compile with musl riscv64 toolchain

# ─── Toolchain ───
CROSS_COMPILE ?= $(HOME)/repo/github/globi/duo-tdl-examples/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-
CC = $(CROSS_COMPILE)gcc
STRIP = $(CROSS_COMPILE)strip

# ─── Middleware paths (from duo-buildroot-sdk-v2) ───
MW_PATH = $(HOME)/repo/github/globi/cvitek-tdl-sdk-sg200x/sample/3rd/middleware/v2
MW_INC = $(MW_PATH)/include
MW_LIB = $(MW_PATH)/lib
MW_3RD_LIB = $(MW_LIB)/3rd
MW_SAMPLE = $(MW_PATH)/sample/common

# ─── Flags ───
CFLAGS = -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d -Os -s \
         -std=gnu11 -Wall -Wno-pointer-to-int-cast -fsigned-char -fno-common \
         -DCV181X -D_MIDDLEWARE_V2_

INCLUDES = -I$(MW_INC) -I$(MW_INC)/linux -I$(MW_INC)/isp/cv181x -I$(MW_SAMPLE)

LDFLAGS = -mcpu=c906fdv

LIBS = -L$(MW_LIB) -L$(MW_3RD_LIB) \
       -lini -lsns_full -lsample -lisp -lvenc -lawb -lae -laf \
       -lcvi_bin -lcvi_bin_isp -lmisc -lisp_algo -lsys -lvpu \
       -lpthread -latomic -ldl

# ─── Common middleware sample objects ───
MW_OBJS = $(MW_SAMPLE)/sample_common_vi.o \
          $(MW_SAMPLE)/sample_common_vpss.o \
          $(MW_SAMPLE)/sample_common_sys.o \
          $(MW_SAMPLE)/sample_common_platform.o \
          $(MW_SAMPLE)/sample_common_isp.o \
          $(MW_SAMPLE)/sample_common_venc.o \
          $(MW_SAMPLE)/sample_common_bin.o \
          $(MW_SAMPLE)/sample_common_sensor.o

TARGET = camera_capture
SRCS = camera_capture.c
OBJS = camera_capture.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) $(MW_OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(MW_OBJS) $(LIBS)
	$(STRIP) $@
	@echo "Build complete: $@"
	@file $@

camera_capture.o: camera_capture.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_vi.o: $(MW_SAMPLE)/sample_common_vi.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_vpss.o: $(MW_SAMPLE)/sample_common_vpss.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_sys.o: $(MW_SAMPLE)/sample_common_sys.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_platform.o: $(MW_SAMPLE)/sample_common_platform.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_isp.o: $(MW_SAMPLE)/sample_common_isp.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_venc.o: $(MW_SAMPLE)/sample_common_venc.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_bin.o: $(MW_SAMPLE)/sample_common_bin.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(MW_SAMPLE)/sample_common_sensor.o: $(MW_SAMPLE)/sample_common_sensor.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
	find $(MW_SAMPLE) -name "*.o" -delete
